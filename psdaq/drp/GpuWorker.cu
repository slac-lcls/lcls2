#include "GpuWorker.hh"

#include "Detector.hh"
#include "spscqueue.hh"
#include "psalg/utils/SysLog.hh"
#include "psdaq/service/EbDgram.hh"
#include "psdaq/service/kwargs.hh"
#include "psdaq/service/range.hh"
#include "psdaq/aes-stream-drivers/DmaDest.h"
#include "psdaq/aes-stream-drivers/GpuAsyncRegs.h"

#include <thread>

using namespace XtcData;
using namespace Pds;
using namespace Drp;
using logging = psalg::SysLog;


namespace Drp {

#ifdef SUDO
static const unsigned GPU_OFFSET      = GPU_ASYNC_CORE_OFFSET;
#endif
static const size_t   DMA_BUFFER_SIZE = 256*1024; // ePixUHR = 192*168*4 ASICs * 2B

//https://github.com/slaclab/surf/blob/main/axi/dma/rtl/v2/AxiStreamDmaV2Write.vhd
struct DmaDsc
{
  uint32_t error;
  uint32_t size;
  uint32_t _rsvd[6];
};

};


MemPoolGpu::MemPoolGpu(Parameters& para) :
  MemPool(para),
  m_setMaskBytesDone(0)
{
  m_dmaCount = MAX_BUFFERS;             // @todo: use a setter method?
  dmaBuffers = nullptr;                 // Unused: cause a crash if accessed

  ////////////////////////////////////////////
  // Setup GPU
  ////////////////////////////////////////////

  if (para.verbose)
    m_context.listDevices();

  unsigned gpuId = 0;
  if (para.kwargs.find("gpuId") != para.kwargs.end())
    gpuId = std::stoul(const_cast<Parameters&>(para).kwargs["gpuId"]);

  if (!m_context.init(gpuId, para.verbose == 0)) {
    logging::critical("CUDA initialize failed");
    abort();
  }
  logging::debug("Done with context setup\n");

  ////////////////////////////////////
  // Setup memory
  ////////////////////////////////////

  unsigned worker = 0;
  std::vector<int> units;
  auto pos = para.device.find("_", 0);
  getRange(para.device.substr(pos+1, para.device.length()), units);
  for (auto unit : units) {
    std::string device(para.device.substr(0, pos+1) + std::to_string(unit));
    m_segs.emplace_back(device);
    logging::info("PGP device '%s' opened", device.c_str());

    auto& seg = m_segs[worker];

    // Clear out any left-overs from last time
    int res = gpuRemNvidiaMemory(seg.gpu.fd());
    if (res < 0)  logging::error("Error in gpuRemNvidiaMemory\n");
    logging::debug("Done with gpuRemNvidiaMemory() cleanup\n");

#ifdef SUDO
    ////////////////////////////////////////////////
    // Map FPGA register space to GPU
    ////////////////////////////////////////////////

    /** Map the GpuAsyncCore FPGA registers **/
    if (gpuMapHostFpgaMem(&seg.swFpgaRegs, seg.gpu.fd(), GPU_OFFSET, 0x100000) < 0) {
      logging::critical("Failed to map GpuAsyncCore at offset=%d, size = %d", GPU_OFFSET, 0x100000);
      abort();
    }

    /** Compute 'write start' register location using the device pointer to GpuAsyncCore **/
    seg.hwWriteStart = seg.swFpgaRegs.dptr + 0x300;
#endif

    logging::debug("Mapped FPGA registers");

    // @todo: Get the DMA size from somewhere - query the device?  No, the device is told below.
    //        Get it from the command line?
    m_dmaSize = DMA_BUFFER_SIZE;

    // Allocate buffers on the GPU
    // This handles allocating buffers on the device and registering them with the driver.
    for (unsigned i = 0; i < dmaCount(); ++i) {
      if (gpuMapFpgaMem(&seg.dmaBuffers[i], seg.gpu.fd(), 0, dmaSize(), 1) != 0) {
        logging::critical("Worker %d failed to alloc buffer list at number %zd", worker, i);
        abort();
      }
#ifdef SUDO
      auto dmaBufAddr0 = readRegister<uint32_t>(seg.swFpgaRegs.ptr, GPU_ASYNC_WR_ADDR(i));
      auto dmaBufAddr1 = readRegister<uint32_t>(seg.swFpgaRegs.ptr, GPU_ASYNC_WR_ADDR(i)+4);
      auto dmaBufSize  = readRegister<uint32_t>(seg.swFpgaRegs.ptr, GPU_ASYNC_WR_SIZE(i));
      uint64_t dmaBufAddr = ((uint64_t)dmaBufAddr1 << 32) | dmaBufAddr0;
      printf("*** DMA buffer[%d] dptr %p, addr %p, size %u\n",
             i, (void*)seg.dmaBuffers[i].dptr, (void*)dmaBufAddr, dmaBufSize);
#endif
    }

    logging::debug("Done with device mem alloc for worker %d\n", worker);
    ++worker;
  }

  logging::debug("Done with setting up memory\n");

  // There is one worker per PGP device
  para.nworkers = m_segs.size();

  // Continue with initialization of the base class
  _initialize(para);
}

MemPoolGpu::~MemPoolGpu()
{
  for (auto& seg : m_segs) {
    for (unsigned i = 0; i < dmaCount(); ++i) {
      gpuUnmapFpgaMem(&seg.dmaBuffers[i]);
    }

    ssize_t rc;
    if ((rc = gpuRemNvidiaMemory(seg.gpu.fd())) < 0)
      logging::error("gpuRemNvidiaMemory failed: %zd: %M", rc);
  }
  m_segs.clear();
}

int MemPoolGpu::setMaskBytes(uint8_t laneMask, unsigned virtChan)
{
    int retval = 0;
    if (m_setMaskBytesDone == m_segs.size()) {
        logging::debug("%s: earlier setting in effect", __PRETTY_FUNCTION__);
    } else {
        uint8_t mask[DMA_MASK_SIZE];
        dmaInitMaskBytes(mask);
        for (unsigned i=0; i<PGP_MAX_LANES; i++) {
            if (laneMask & (1 << i)) {
                uint32_t channel = i;
                uint32_t dest = dmaDest(channel, virtChan);
                logging::info("setting lane  %u, dest 0x%x", i, dest);
                dmaAddMaskBytes(mask, dest);
            }
        }
        for (const auto& seg : m_segs) {
            if (dmaSetMaskBytes(seg.gpu.fd(), mask)) {
                retval = 1; // error
            } else {
                ++m_setMaskBytesDone;
            }
        }
    }
    return retval;
}


GpuWorker::GpuWorker(unsigned worker, const Parameters& para, MemPoolGpu& pool) :
  m_pool       (pool),
  m_terminate_h(false),
  m_dmaQueue   (m_pool.nbuffers()),     // @todo: Revisit depth
  m_worker     (worker),
  m_para       (para)
{
  ////////////////////////////////////
  // Allocate streams
  ////////////////////////////////////

  /** Allocate a stream per buffer **/
  m_streams.resize(m_pool.dmaCount());
  for (auto& stream : m_streams) {
    chkFatal(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
  logging::debug("Done with creating streams\n");

  // Set up thread termination flag in managed memory
  chkError(cudaMallocManaged(&m_terminate_d, sizeof(*m_terminate_d)));
  *m_terminate_d = 0;

  // Set up a done flag to cache m_terminate's value and avoid some PCIe transactions
  chkError(cudaMalloc(&m_done, sizeof(*m_done)));
  chkError(cudaMemset(m_done, 0, sizeof(*m_done)));

#ifdef SUDO
  //// Set up buffer-ready flags
  //for (unsigned i = 0; i < m_pool.dmaCount(); ++i) {
  //  chkError(cudaMallocManaged(&m_bufRdy[i], m_pool.dmaCount() * sizeof(*m_bufRdy[i])));
  //  *m_bufRdy[i] = 1;                   // Declare buffers initially ready
  //}

  unsigned nBufs = m_pool.nbuffers();
  m_ringIndex_h = new Gpu::RingIndex(nBufs, m_pool.dmaCount(), m_terminate_h, *m_terminate_d);
  chkError(cudaMalloc(&m_ringIndex_d, sizeof(*m_ringIndex_d)));
  chkError(cudaMemcpy(m_ringIndex_d, m_ringIndex_h, sizeof(*m_ringIndex_d), cudaMemcpyHostToDevice));

  for (unsigned i = 0; i < m_pool.dmaCount(); ++i) {
    chkError(cudaMalloc(&m_head[i], sizeof(*m_head[i])));
    chkError(cudaMemset(m_head[i], 0, sizeof(*m_head[i])));
    chkError(cudaMalloc(&m_rdyCtr[i], sizeof(*m_rdyCtr[i])));
    chkError(cudaMemset(m_rdyCtr[i], 0, sizeof(*m_rdyCtr[i])));
    chkError(cudaMalloc(&m_hostWriteBuf[i], sizeof(*m_hostWriteBuf[i])));
    chkError(cudaMemset(m_hostWriteBuf[i], 0, sizeof(*m_hostWriteBuf[i])));
  }

  m_hostWriteBufs.resize(m_ringIndex_h->size());
  size_t size = sizeof(DmaDsc)+sizeof(TimingHeader); //m_pool.dmaSize();
  chkError(cudaMalloc(&m_hostWriteBufs_d, m_ringIndex_h->size() * sizeof(*m_hostWriteBufs_d)));
  for (auto& hostWriteBuf : m_hostWriteBufs) {
    chkError(cudaMallocManaged(&hostWriteBuf, size));
    chkError(cudaMemset(hostWriteBuf, 0, size)); // Avoid rereading junk on re-Configure
  }
  chkError(cudaMemcpy(m_hostWriteBufs_d, m_hostWriteBufs.data(), m_hostWriteBufs.size() * sizeof(m_hostWriteBufs[0]), cudaMemcpyHostToDevice));
#else
  m_hostWriteBufs.resize(m_pool.dmaCount());
#endif

  // Prepare the CUDA graphs
  const auto& seg = m_pool.segs()[worker];
  m_graphs.resize(m_pool.dmaCount());
  m_graphExecs.resize(m_pool.dmaCount());
  for (unsigned i = 0; i < m_pool.dmaCount(); ++i) {
    if (_setupCudaGraphs(seg, i)) {
      logging::critical("Failed to set up CUDA graphs");
      abort();
    }
  }
}

GpuWorker::~GpuWorker()
{
  for (auto& stream : m_streams) {
    chkError(cudaStreamDestroy(stream));
  }

  for (auto& hostWriteBuf : m_hostWriteBufs) {
    chkError(cudaFree(hostWriteBuf));
  }
#ifdef SUDO
  chkError(cudaFree(m_hostWriteBufs_d));
  for (unsigned i = 0; i < m_pool.dmaCount(); ++i) {
  //  chkError(cudaFree(m_bufRdy[i]));
    chkError(cudaFree(m_hostWriteBuf[i]));
    chkError(cudaFree(m_head[i]));
    chkError(cudaFree(m_rdyCtr[i]));
  }
  delete m_ringIndex_h;
#endif
  chkError(cudaFree(m_terminate_d));
  chkError(cudaFree(m_done));
}

int GpuWorker::_setupCudaGraphs(const DetSeg& seg, int instance)
{
  // Generate the graph
  if (m_graphs[instance] == 0) {
    logging::debug("Recording graph %d of CUDA execution", instance);
#ifndef SUDO
    size_t size = sizeof(DmaDsc)+sizeof(TimingHeader); //m_pool.dmaSize();
    chkError(cudaMallocManaged(&m_hostWriteBufs[instance], size));
    chkError(cudaMemset(m_hostWriteBufs[instance], 0, size)); // Avoid rereading junk on re-Configure
    chkError(cudaStreamAttachMemAsync(m_streams[instance], m_hostWriteBufs[instance], 0, cudaMemAttachHost));
    m_graphs[instance] = _recordGraph(m_streams[instance], seg.dmaBuffers[instance].dptr, m_hostWriteBufs[instance]);
#else
    for (unsigned i = 0; i < m_ringIndex_h->size(); ++i) {
      chkError(cudaStreamAttachMemAsync(m_streams[instance], m_hostWriteBufs[i], 0, cudaMemAttachHost));
    }
    m_graphs[instance] = _recordGraph(m_streams[instance], seg.dmaBuffers[instance].dptr, seg.hwWriteStart);
#endif
    if (m_graphs[instance] == 0)
      return -1;
  }

  // Instantiate the graph. The resulting CUgraphExec may only be executed once
  // at any given time.  I believe it can be reused, but it cannot be launched
  // while it is already running.  If we wanted to launch multiple, we would
  // instantiate multiple CUgraphExec's and then launch those individually.
  if (chkError(cudaGraphInstantiate(&m_graphExecs[instance], m_graphs[instance], cudaGraphInstantiateFlagDeviceLaunch),
               "Graph create failed")) {
    return -1;
  }

  // Upload the graph so it can be launched by the scheduler kernel later
  logging::debug("Uploading graph...");
  if (chkError(cudaGraphUpload(m_graphExecs[instance], m_streams[instance]), "Graph upload failed")) {
    return -1;
  }

  return 0;
}

static __global__ void _waitValueGEQ(const volatile uint32_t* mem,
                                     uint32_t                 val,
#ifdef SUDO
                                     //cuda::atomic<int>*       bufRdy,
                                     unsigned                 instance,
                                     Gpu::RingIndex&          ringIndex,
                                     uint32_t**               outBufs,
                                     uint32_t**               out,
                                     unsigned*                head,
                                     cuda::atomic<unsigned, cuda::thread_scope_block>& rdyCtr,
#endif
                                     const cuda::atomic<int>* terminate,
                                     bool*                    done)
{
#ifdef SUDO
  //// Wait for the host buffer to become ready
  //while (!*bufRdy) {
  //  if (terminate->load(cuda::memory_order_acquire)) {
  //    *done = true;
  //    return;
  //  }
  //}
  //bufRdy->store(0, cuda::memory_order_release); // Mark it not ready

  auto idx = ringIndex.prepare(instance);
  *out = outBufs[idx];
  *head = idx;
  rdyCtr.store(0, cuda::memory_order_release);
#endif

  // Wait for data to be DMAed
  while (*mem < val) {
    if (terminate->load(cuda::memory_order_acquire)) {
      *done = true;
      return;
    }
  }
}

#ifndef SUDO
static __global__ void _event(uint32_t* out, uint32_t* in, const bool& done)
#else
static __global__ void _event(uint32_t* const* __restrict__                     pOut,
                              uint32_t* const  __restrict__                     in,
                              cuda::atomic<unsigned, cuda::thread_scope_block>& rdyCtr,
                              Gpu::RingIndex&                                   ringIndex,
                              const unsigned&                                   head,
                              const bool&                                       done)
#endif
{
#ifdef SUDO
  uint32_t* const __restrict__ out = *pOut;
#endif
  if (done)  return;

  const DmaDsc* const __restrict__ dmaDsc = (const DmaDsc*)in;

  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= (sizeof(*dmaDsc)+dmaDsc->size)/sizeof(*out))
    return;

  const auto nWords = (sizeof(DmaDsc)+sizeof(TimingHeader))/sizeof(*out);
  if (offset < nWords) {
    out[offset] = in[offset];           // Each of these is a PCIe transaction
#ifdef SUDO
    if (++rdyCtr == nWords) {
      ringIndex.produce(head);
    }
#endif
  }

  // Clear the GPU memory handshake space to zero
  //   Do this _after_ we captured the size from the dmaDsc!
  if (offset == 1) {
    in[offset] = 0;
  }

  // @todo: Need a __device__ version of TimingHeader
  //const TimingHeader* timingHeader = (const TimingHeader*)&dmaDsc[1];
  auto svc = (in[8] >> 24) & 0xf;
  if (svc != TransitionId::L1Accept)  return;

  // @todo: Process the data to extract TEB input and calibrate.  Also launch reduce/compress algo?
  //det.event(*(TimingHeader*)&in[8], head);
}

// This will re-launch the current graph
static __global__ void _graphLoop(const bool& done)
{
  if (done)  return;

  cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
}

cudaGraph_t GpuWorker::_recordGraph(cudaStream_t& stream,
                                    CUdeviceptr   dmaBuffer,
#ifndef SUDO
                                    uint32_t*     hostWriteBuf
#else
                                    CUdeviceptr   hwWriteStart
#endif
                                    )
{
  int instance = &stream - &m_streams[0];

  if (chkError(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "Stream capture failed")) {
    return 0;
  }

#ifdef SUDO
  /****************************************************************************
   * Clear the handshake space
   * Originally was cuStreamWriteValue32, but the stream functions are not
   * supported within graphs. cuMemsetD32Async acts as a good replacement.
   ****************************************************************************/
  // This is now done in the event CUDA kernel
  //chkError(cuMemsetD32Async(dmaBuffer + 4, 0, 1, stream)); // Formerly cuStreamWriteValue32

  // Write to the DMA start register in the FPGA to trigger the write
  chkError(cuMemsetD8Async(hwWriteStart + 4 * instance, 1, 1, stream)); // Formerly cuStreamWriteValue32
#endif

  /*****************************************************************************
   * Spin on the handshake location until the value is >= to 1
   * This waits for the data to arrive before starting the processing
   * Originally this was a call to cuStreamWait, but that is not supported by
   * graphs, so instead we use a waitForGEQ kernel to spin on the location
   * until data is ready to be processed.
   * @todo: This may have negative implications on GPU scheduling.
   *        Need to profile!!!
   ****************************************************************************/
#ifndef SUDO
  _waitValueGEQ<<<1, 1, 1, stream>>>((uint32_t*)(dmaBuffer + 4), 0x1, m_terminate_d, m_done);
#else
  uint32_t bufMask = m_pool.nbuffers() - 1;
  //_waitValueGEQ<<<1, 1, 1, stream>>>((uint32_t*)(dmaBuffer + 4), 0x1, m_bufRdy[instance], m_terminate_d, m_done);
  _waitValueGEQ<<<1, 1, 1, stream>>>((uint32_t*)(dmaBuffer + 4),
                                     0x1, // GEQ comparison value
                                     instance,
                                     *m_ringIndex_d,
                                     m_hostWriteBufs_d,
                                     (uint32_t**)m_hostWriteBuf[instance],
                                     m_head[instance],
                                     *m_rdyCtr[instance],
                                     m_terminate_d,
                                     m_done);
#endif

  // An alternative to the above kernel is to do the waiting on the CPU instead...
  //chkError(cuLaunchHostFunc(stream, check_memory, (void*)buffer));

  // Do GPU processing here, this simply copies data from the write buffer to the read buffer
  auto blkCnt = m_pool.dmaSize() >> (10 + 2); // +2 because each GPU thread handles a uint32_t
  if (blkCnt == 0)  blkCnt = 1;               // @todo:abort?
#ifndef SUDO
  _event<<<blkCnt, 1024, 1, stream>>>((uint32_t*)hostWriteBuf, (uint32_t*)dmaBuffer, *m_done);
#else
  _event<<<blkCnt, 1024, 1, stream>>>((uint32_t**)m_hostWriteBuf[instance],
                                      (uint32_t*)dmaBuffer,
                                      *m_rdyCtr[instance],
                                      *m_ringIndex_d,
                                      *m_head[instance],
                                      *m_done);
#endif

  // Re-launch! Additional behavior can be put in graphLoop as needed. For now, it just re-launches the current graph.
  _graphLoop<<<1, 1, 0, stream>>>(*m_done);

  cudaGraph_t graph;
  if (chkError(cudaStreamEndCapture(stream, &graph), "Stream capture failed")) {
    return 0;
  }

  return graph;
}

DmaTgt_t GpuWorker::dmaTarget() const
{
  // @todo: This line addresses only lane 0
  return dmaTgtGet(m_pool.segs()[m_worker].gpu);
}

void GpuWorker::dmaTarget(DmaTgt_t dest)
{
  // @todo: This line addresses only lane 0
  dmaTgtSet(m_pool.segs()[m_worker].gpu, dest);
}

void GpuWorker::start(Detector* det, GpuMetrics& metrics)
{
  m_terminate_h.store(false, std::memory_order_release);

  // Launch the Reader thread
  m_thread = std::thread(&GpuWorker::_reader, std::ref(*this),
                         std::ref(*det), std::ref(metrics));
}

void GpuWorker::stop()
{
  logging::warning("GpuWorker::stop called for worker %d\n", m_worker);

  chkError(cuCtxSetCurrent(m_pool.context().context()));

  // Stop and clean up the threads
  m_terminate_h.store(true, std::memory_order_release);
  m_thread.join();
}

void GpuWorker::freeDma(unsigned index)
{
#ifndef SUDO
  // Write to the DMA start register in the FPGA
  //logging::debug("Trigger write to buffer %d\n", dmaIdx);
  const auto& seg = m_pool.segs()[m_worker];
  auto rc = gpuSetWriteEn(seg.gpu.fd(), index);
  if (rc < 0) {
    logging::critical("Failed to reenable buffer %d for write: %zd: %m", index, rc);
    abort();
  }
#else
  //m_bufRdy[index]->store(1, cuda::memory_order_release); // Mark the buffer available
  m_ringIndex_h->release(index);
#endif

  m_pool.freeDma(1, nullptr);
}

void GpuWorker::handleBrokenEvent(const PGPEvent&)
{
}

void GpuWorker::resetEventCounter()
{
}

TimingHeader* GpuWorker::timingHeader(unsigned index) const
{
  return (TimingHeader*)&(m_hostWriteBufs[index])[8];
}

void GpuWorker::_reader(Detector& det, GpuMetrics& metrics)
{
  logging::info("GpuWorker[%d] starting\n", m_worker);
  chkError(cuCtxSetCurrent(m_pool.context().context()));  // Needed, else kernels misbehave

  // Ensure that timing messages are DMAed to the GPU
  dmaTarget(DmaTgt_t::GPU);

  // Ensure that the DMA round-robin index starts with buffer 0
  const auto& seg = m_pool.segs()[m_worker];
  dmaIdxReset(seg.gpu);

  resetEventCounter();

  for (unsigned dmaIdx = 0; dmaIdx < m_pool.dmaCount(); ++dmaIdx) {
    // Clear the GPU memory handshake space to zero
    auto dmaBuffer = seg.dmaBuffers[dmaIdx].dptr;
    chkFatal(cuStreamWriteValue32(m_streams[dmaIdx], dmaBuffer + 4, 0x00, 0));
    chkError(cudaStreamSynchronize(m_streams[dmaIdx]));

    // Write to the DMA start register in the FPGA
    auto rc = gpuSetWriteEn(seg.gpu.fd(), dmaIdx);
    if (rc < 0) {
      logging::critical("Failed to reenable buffer %d for write: %zd: %m", dmaIdx, rc);
      abort();
    }
  }
  for (unsigned dmaIdx = 0; dmaIdx < m_pool.dmaCount(); ++dmaIdx) {
    chkFatal(cudaGraphLaunch(m_graphExecs[dmaIdx], m_streams[dmaIdx]));
  }

#ifndef SUDO
  const uint32_t bufferMask = m_pool.nDmaBuffers() - 1;
#else
  const uint32_t bufferMask = m_ringIndex_h->size() - 1;
  unsigned tail = 0;
  unsigned head = 0;
#endif

  uint64_t lastPid = 0;
  while (!m_terminate_h.load(std::memory_order_acquire)) {
#ifndef SUDO
    for (unsigned index = 0; index < m_pool.dmaCount(); ++index) {
      unsigned nDmaRet = 1;
#else
    tail = head;
    head = m_ringIndex_h->consume();
    unsigned index = tail;
    while (index != head) {
      unsigned nDmaRet = head - index;
#endif
      const volatile auto dsc = (DmaDsc*)(m_hostWriteBufs[index]);
      const volatile auto th  = (TimingHeader*)&dsc[1];

      metrics.m_nDmaRet.store(0);
      // Wait for the GPU to have processed an event
      //chkFatal(cuCtxSynchronize());
      //chkError(cuStreamSynchronize(m_streams[index]));
      uint64_t pid;
      while (!m_terminate_h.load(std::memory_order_acquire)) {
        pid = th->pulseId();
        if (pid > lastPid)  break;
        if (!lastPid && !pid)  break;   // Expect lastPid to be 0 only on startup
      }
      if (m_terminate_h.load(std::memory_order_acquire))  break;
      if (!pid)  continue;              // Search for a DMA buffer with data in it
      lastPid = pid;

      metrics.m_nDmaRet.store(nDmaRet);
      m_pool.allocateDma();

      logging::debug("*** dma %d hdr: err %08x,  sz %08x, rsvd %08x %08x %08x %08x %08x %08x\n",
                     index, dsc->error, dsc->size, dsc->_rsvd[0], dsc->_rsvd[1], dsc->_rsvd[2],
                     dsc->_rsvd[3], dsc->_rsvd[4], dsc->_rsvd[5]);
      logging::debug("**G dma %d  th: ctl %02x, pid %014lx, ts %016lx, env %08x, ctr %08x, opq %08x %08x\n",
                     index, th->control(), th->pulseId(), th->time.value(), th->env, th->evtCounter,
                     th->_opaque[0], th->_opaque[1]);

      uint32_t size = dsc->size;
      ///uint32_t lane = 0;                  // The lane is always 0 for GPU-enabled PGP devices
      metrics.m_dmaSize   = size;
      metrics.m_dmaBytes += size;
      // @todo: Is this the case here also?
      // dmaReadBulkIndex() returns a maximum size of m_pool.dmaSize(), never larger.
      // If the DMA overflowed the buffer, the excess is returned in a 2nd DMA buffer,
      // which thus won't have the expected header.  Take the exact match as an overflow indicator.
      if (size == m_pool.dmaSize()) {
        logging::critical("%d DMA overflowed buffer: %d vs %d", index, size, m_pool.dmaSize());
        //abort();
      }

      const Pds::TimingHeader* timingHeader = th; //det.getTimingHeader(index);
      if (!timingHeader)  printf("*** No timingHeader at ctr %d\n", th->evtCounter);
      ///uint32_t evtCounter = timingHeader->evtCounter & 0xffffff;
      ///unsigned pgpIndex = evtCounter & bufferMask;
      ///PGPEvent*  event  = &m_pool.pgpEvents[pgpIndex];
      ///if (!event)  printf("*** No pgpEvent for ctr %d\n", pgpIndex);
      ///DmaBuffer* buffer = &event->buffers[lane]; // @todo: Do we care about this?
      ///if (!buffer)  printf("*** No pgpEvent.buffer for lane %d, ctr %d\n", lane, pgpIndex);
      ///buffer->size = size;                       //   "
      ///buffer->index = index;                     //   "
      ///event->mask |= (1 << lane);

      if (dsc->error) {
        logging::error("DMA error 0x%x",dsc->error);
        ///handleBrokenEvent(*event);
        freeDma(index);                 // Leaves event mask = 0
        metrics.m_nDmaErrors += 1;
        continue;
      }

      if (timingHeader->error()) {
        logging::error("Timing header error bit is set");
        metrics.m_nTmgHdrError += 1;
      }
      XtcData::TransitionId::Value transitionId = timingHeader->service();

      auto rogs = timingHeader->readoutGroups();
      if ((rogs & (1 << m_para.partition)) == 0) {
        logging::debug("%s @ %u.%09u (%014lx) without common readout group (%u) in env 0x%08x",
                       XtcData::TransitionId::name(transitionId),
                       timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                       timingHeader->pulseId(), m_para.partition, timingHeader->env);
        ///handleBrokenEvent(*event);
        freeDma(index);                 // Leaves event mask = 0
        metrics.m_nNoComRoG += 1;
        continue;
      }

      // @todo: Shouldn't this check for missing RoGs on all transitions?
      if (transitionId == XtcData::TransitionId::SlowUpdate) {
        uint16_t missingRogs = m_para.rogMask & ~rogs;
        if (missingRogs) {
          logging::debug("%s @ %u.%09u (%014lx) missing readout group(s) (0x%04x) in env 0x%08x",
                         XtcData::TransitionId::name(transitionId),
                         timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                         timingHeader->pulseId(), missingRogs, timingHeader->env);
          ///handleBrokenEvent(*event);
          freeDma(index);             // Leaves event mask = 0
          metrics.m_nMissingRoGs += 1;
          continue;
        }
      }

      const uint32_t* data = reinterpret_cast<const uint32_t*>(th);
      logging::debug("GpuWorker  size %u  hdr %016lx.%016lx.%08x  err 0x%x",
                     size,
                     reinterpret_cast<const uint64_t*>(data)[0], // PulseId
                     reinterpret_cast<const uint64_t*>(data)[1], // Timestamp
                     reinterpret_cast<const uint32_t*>(data)[4], // env
                     dsc->error);

      if (transitionId != XtcData::TransitionId::L1Accept) {
        if (transitionId != XtcData::TransitionId::SlowUpdate) {
          logging::info("GpuWorker  saw %s @ %u.%09u (%014lx)",
                        XtcData::TransitionId::name(transitionId),
                        timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                        timingHeader->pulseId());
        }
        else {
          logging::debug("GpuWorker  saw %s @ %u.%09u (%014lx)",
                         XtcData::TransitionId::name(transitionId),
                         timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                         timingHeader->pulseId());
        }
        if (transitionId == XtcData::TransitionId::BeginRun) {
          resetEventCounter();
        }
      }

      metrics.m_nevents += 1;
      m_dmaQueue.push(index);
#ifdef SUDO
      index = (index+1)&bufferMask;
#endif
    }
  }

  m_dmaQueue.shutdown();

  logging::info("Shutting down GPU streams");
  // Not atomic, but what can you do?
  chkError(cudaMemset(m_terminate_d, 1, sizeof(*m_terminate_d))); // 0x01010101, oh well, still != 0

  logging::debug("GpuWorker[%d] exiting\n", m_worker);
}
