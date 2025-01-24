//#define SUDO

#include "GpuWorker.hh"

#include "Detector.hh"
#include "spscqueue.hh"
#include "psalg/utils/SysLog.hh"
#include "psdaq/service/EbDgram.hh"
#include "psdaq/service/kwargs.hh"
#include "psdaq/service/range.hh"
#include "psdaq/aes-stream-drivers/DmaDest.h"

#include <thread>

using namespace XtcData;
using namespace Pds;
using namespace Drp;
using logging = psalg::SysLog;


namespace Drp {

#ifdef SUDO
static const unsigned GPU_OFFSET      = 0x28000;
#endif
static const size_t   DMA_BUFFER_SIZE = 128*1024;

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

  if (!m_context.init(gpuId)) {
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
  m_segs.resize(units.size());
  for (auto unit : units) {
    auto& seg = m_segs[worker];

    std::string device(para.device.substr(0, pos+1) + std::to_string(unit));
    seg.fd = open(device.c_str(), O_RDWR);
    if (seg.fd < 0) {
      logging::critical("Error opening %s: %m", device.c_str());
      abort();
    }
    logging::info("PGP device '%s' opened", device.c_str());

    // Clear out any left-overs from last time
    int res = gpuRemNvidiaMemory(seg.fd);
    if (res < 0)  logging::error("Error in gpuRemNvidiaMemory\n");
    logging::debug("Done with gpuRemNvidiaMemory() cleanup\n");

    ////////////////////////////////////////////////
    // Map FPGA register space to GPU
    ////////////////////////////////////////////////

#ifdef SUDO
    /** Map the GpuAsyncCore FPGA registers **/
    if (gpuMapHostFpgaMem(&seg.swFpgaRegs, seg.fd, GPU_OFFSET, 0x100000) < 0) {
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
      if (gpuMapFpgaMem(&seg.dmaBuffers[i], seg.fd, 0, dmaSize(), 1) != 0) {
        logging::critical("Worker %d failed to alloc buffer list at number %zd", worker, i);
        abort();
      }
      //auto dmaBufAddr = readRegister<void*>   (seg.swFpgaRegs.ptr, GPU_ASYNC_WR_ADDR(i));
      //auto dmaBufSize = readRegister<uint32_t>(seg.swFpgaRegs.ptr, GPU_ASYNC_WR_SIZE(i));
      //printf("*** DMA buffer[%d] addr %p, size %u\n", i, dmaBufAddr, dmaBufSize);
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
  for (auto seg : m_segs) {
    for (unsigned i = 0; i < dmaCount(); ++i) {
      gpuUnmapFpgaMem(&seg.dmaBuffers[i]);
    }

    ssize_t rc;
    if ((rc = gpuRemNvidiaMemory(seg.fd)) < 0)
      logging::error("gpuRemNvidiaMemory failed: %zd: %M", rc);

    close(seg.fd);
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
        for (auto seg : m_segs) {
            if (dmaSetMaskBytes(seg.fd, mask)) {
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
  m_seg        (m_pool.segs()[worker]),
  m_terminate_h(false),
  m_dmaQueue   (4),                     // @todo: Revisit depth
  ///m_batchStart(1),
  ///m_batchSize (0),
  m_worker     (worker),
  m_para       (para)
{
  printf("*** GpuWorker::ctor for #%d\n", worker);

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

  // Set up buffer-ready flags
  //for (unsigned i = 0; i < m_pool.dmaCount(); ++i) {
  //  chkError(cudaMallocManaged(&m_bufRdy[i], m_pool.dmaCount() * sizeof(*m_bufRdy[i])));
  //  chkError(cudaStreamAttachMemAsync(m_streams[i], m_bufRdy[i], 0, cudaMemAttachHost));
  //  *m_bufRdy[i] = 0;
  //}

  // Prepare the CUDA graphs
  auto& seg = m_pool.segs()[worker];
  m_graphs.resize(m_pool.dmaCount());
  m_graphExecs.resize(m_pool.dmaCount());
  m_hostWriteBufs.resize(m_pool.dmaCount());
  for (unsigned i = 0; i < m_pool.dmaCount(); ++i) {
    if (_setupCudaGraphs(seg, i)) {
      logging::critical("Failed to set up CUDA graphs");
      abort();
    }
  }
}

GpuWorker::~GpuWorker()
{
  printf("*** GpuWorker::dtor for #%d\n", m_worker);

  for (auto& stream : m_streams) {
    chkError(cudaStreamDestroy(stream));
  }

  for (unsigned i = 0; i < m_pool.dmaCount(); ++i) {
    chkError(cudaFree(m_hostWriteBufs[i]));
    //cudaFree(m_bufRdy[i]);
  }
  chkError(cudaFree(m_terminate_d));
  chkError(cudaFree(m_done));
}

int GpuWorker::_setupCudaGraphs(const DetSeg& seg, int instance)
{
  // Generate the graph
  if (m_graphs[instance] == 0) {
    logging::debug("Recording graph %d of CUDA execution", instance);
    size_t size = sizeof(DmaDsc)+sizeof(TimingHeader); //m_pool.dmaSize();
    chkError(cudaMallocManaged(&m_hostWriteBufs[instance], size));
    chkError(cudaMemset(m_hostWriteBufs[instance], 0, size)); // Avoid rereading junk on re-Configure
    chkError(cudaStreamAttachMemAsync(m_streams[instance], m_hostWriteBufs[instance], 0, cudaMemAttachHost));
    m_graphs[instance] = _recordGraph(m_streams[instance], seg.dmaBuffers[instance].dptr, seg.hwWriteStart, m_hostWriteBufs[instance]); //, m_bufRdy[instance]);
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

__global__ void waitValueGEQ(const volatile uint32_t* mem,
                             uint32_t                 val,
                             //const volatile int*      bufRdy,
                             const cuda::atomic<int>* terminate,
                             bool*                    done)
{
  while (*mem < val) { // && *bufRdy);
    if (terminate->load(cuda::memory_order_acquire)) {
      *done = true;
      break;
    }
  }
  //printf("*** GEQ %d >= %d\n", *mem, val);
  //*bufRdy = 0;
  //cuda::atomic_ref{*bufRdy}.store(0, cuda::memory_order_release);
}

__global__ void event(uint32_t* out, uint32_t* in, size_t size, const bool& done)
{
  if (done)  return;

  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < size/sizeof(*out)) {
    out[offset] = in[offset];
  }

  // Clear the GPU memory handshake space to zero
  //   Do this _after_ we captured the size from the dmaDsc!
  if (offset == 1) {
    in[offset] = 0;
  }

  // @todo: Process the data to extract TEB input and calibrate.  Also reduce/compress?
  //if (timingHeader->service() == TransitionId::L1Accept)
  //  det.event(*th, event);
  //else if (timingHeader->service() == TransitionId::SlowUpdate)
  //  det.slowUpdate(*th);
}

// This will re-launch the current graph
__global__ void graphLoop(const bool& done)
{
  if (done)  return;

  cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
}

cudaGraph_t GpuWorker::_recordGraph(cudaStream_t& stream,
                                    CUdeviceptr   dmaBuffer,
                                    CUdeviceptr   hwWriteStart,
                                    uint32_t*     hostWriteBuf) //,
                                    //int*          bufRdy)
{
  int instance = &stream - &m_streams[0];

  if (chkError(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "Stream capture failed")) {
    return 0;
  }

  /****************************************************************************
   * Clear the handshake space
   * Originally was cuStreamWriteValue32, but the stream functions are not
   * supported within graphs. cuMemsetD32Async acts as a good replacement.
   ****************************************************************************/
  //chkError(cuMemsetD32Async(dmaBuffer + 4, 0, 1, stream)); // Formerly cuStreamWriteValue32

#ifdef SUDO
  // Write to the DMA start register in the FPGA to trigger the write
  chkError(cuMemsetD8Async(hwWriteStart + 4  * instance, 1, 1, stream)); // Formerly cuStreamWriteValue32
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
  waitValueGEQ<<<1, 1, 1, stream>>>((uint32_t*)(dmaBuffer + 4), 0x1, /*bufRdy,*/ m_terminate_d, m_done);

  // An alternative to the above kernel is to do the waiting on the CPU instead...
  //chkError(cuLaunchHostFunc(stream, check_memory, (void*)buffer));

  // Do GPU processing here, this simply copies data from the write buffer to the read buffer
  event<<<4, 1024, 1, stream>>>((uint32_t*)hostWriteBuf, (uint32_t*)dmaBuffer, sizeof(DmaDsc)+sizeof(TimingHeader), *m_done);

  // Re-launch! Additional behavior can be put in graphLoop as needed. For now, it just re-launches the current graph.
  graphLoop<<<1, 1, 0, stream>>>(*m_done);

  cudaGraph_t graph;
  if (chkError(cudaStreamEndCapture(stream, &graph), "Stream capture failed")) {
    return 0;
  }

  return graph;
}

DmaTgt_t GpuWorker::dmaTarget() const
{
  // @todo: This line addresses only lane 0
  return dmaTgtGet(m_pool.segs()[m_worker].fd);
}

void GpuWorker::dmaTarget(DmaTgt_t dest)
{
  // @todo: This line addresses only lane 0
  dmaTgtSet(m_pool.segs()[m_worker].fd, dest);
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

void GpuWorker::freeDma(unsigned dmaIdx)
{
  m_pool.freeDma(1, nullptr);

  // Write to the DMA start register in the FPGA
  //logging::debug("Trigger write to buffer %d\n", dmaIdx);
#ifndef SUDO
  auto rc = gpuSetWriteEn(m_seg.fd, dmaIdx);
  if (rc < 0) {
    logging::critical("Failed to reenable buffer %d for write: %zd: %m", dmaIdx, rc);
    abort();
  }
#else
  //chkError(cuStreamWriteValue32(m_streams[dmaIdx], m_seg.hwWriteStart + 4 * dmaIdx, 1, 0));
  //*(m_bufRdy[dmaIdx]) = 1;
  cuda::atomic_ref{*(m_bufRdy[dmaIdx])}.store(1, cuda::memory_order_release);
#endif
}

void GpuWorker::handleBrokenEvent(const PGPEvent&)
{
  ///m_batchSize += 1; // Broken events must be included in the batch since f/w advanced evtCounter
}

void GpuWorker::resetEventCounter()
{
  ///m_batchStart = 1;
  ///m_batchSize = 0;
  ///m_batchLast = 0;
}

TimingHeader* GpuWorker::timingHeader(unsigned dmaIdx) const
{
  return (TimingHeader*)&(m_hostWriteBufs[dmaIdx])[8];
}

// @todo: Do we still want these?
//// @todo: This method is called when it has been recognized that data
////        has been DMAed into GPU memory and is ready to be processed
//void GpuWorker::process(Batch& batch, bool& sawDisable)
//{
//  // Set up a buffer pool for timing headers visible to the host
//  // memcpy timing headers from device into this host pool
//  // memcpy the TEB input data right after the timing header?
//  //   Or put them in a separate pool?
//  // Form a batch of them
//  // Return from this routine when batch is full or Disable is seen
//
//}

// @todo: This method is called to wait for data to be DMAed into GPU memory
//        This method must then do several things:
//        - Find and copy the TimingHeader to a host buffer and share that
//          buffer's index
//        - Do the equivalent of the det.event() and det.slowUpdate() routines
//          to reorganize the data and prepare the Xtc header
//        - Prepare the TEB input data

void GpuWorker::_reader(Detector& det, GpuMetrics& metrics)
{
  logging::info("GpuWorker[%d] starting\n", m_worker);
  chkError(cuCtxSetCurrent(m_pool.context().context()));  // Needed, else kernels misbehave

  dmaTarget(DmaTgt_t::GPU); // Ensure that timing messages are DMAed to the GPU

  resetEventCounter();

  for (unsigned dmaIdx = 0; dmaIdx < m_pool.dmaCount(); ++dmaIdx) {
    // Clear the GPU memory handshake space to zero
    auto dmaBuffer = m_seg.dmaBuffers[dmaIdx].dptr;
    chkFatal(cuStreamWriteValue32(m_streams[dmaIdx], dmaBuffer + 4, 0x00, 0));
    chkError(cudaStreamSynchronize(m_streams[dmaIdx]));

    // Write to the DMA start register in the FPGA
    auto rc = gpuSetWriteEn(m_seg.fd, dmaIdx);
    if (rc < 0) {
      logging::critical("Failed to reenable buffer %d for write: %zd: %m", dmaIdx, rc);
      abort();
    }

    chkFatal(cudaGraphLaunch(m_graphExecs[dmaIdx], m_streams[dmaIdx]));
  }

  const uint32_t bufferMask = m_pool.nDmaBuffers() - 1;

  ///bool     flushBatch = false;
  uint64_t lastPid = 0;
  ///unsigned lastPblIndex = 0;
  bool done = false;
  while (!done) {
    for (unsigned dmaIdx = 0; dmaIdx < m_pool.dmaCount(); ++dmaIdx) {
      const volatile auto dsc = (DmaDsc*)(m_hostWriteBufs[dmaIdx]);
      const volatile auto th  = (TimingHeader*)&(m_hostWriteBufs[dmaIdx])[8];

      metrics.m_nDmaRet.store(0);
      // Wait for the GPU to have processed an event
      //chkFatal(cuCtxSynchronize());
      //chkError(cuStreamSynchronize(m_streams[dmaIdx]));
      uint64_t pid;
      while (true) {
        if (m_terminate_h.load(std::memory_order_acquire)) {
          done = true;
          break;
        }

        pid = th->pulseId();
        if (pid > lastPid)  break;
        if (!lastPid && !pid)  break;   // Expect lastPid to be 0 only on startup
      }
      if (done)  break;
      if (!pid)  continue;              // Search for a DMA buffer with data in it
      lastPid = pid;

      unsigned nDmaRet = 1;
      metrics.m_nDmaRet.store(nDmaRet);
      m_pool.allocateDma();

      logging::debug("*** dma %d hdr: err %08x,  sz %08x, rsvd %08x %08x %08x %08x %08x %08x\n",
                     dmaIdx, dsc->error, dsc->size, dsc->_rsvd[0], dsc->_rsvd[1], dsc->_rsvd[2],
                     dsc->_rsvd[3], dsc->_rsvd[4], dsc->_rsvd[5]);
      logging::debug("**G dma %d  th: ctl %02x, pid %014lx, ts %016lx, env %08x, ctr %08x, opq %08x %08x\n",
                     dmaIdx, th->control(), th->pulseId(), th->time.value(), th->env, th->evtCounter,
                     th->_opaque[0], th->_opaque[1]);

      uint32_t size = dsc->size;
      ///uint32_t lane = 0;                  // The lane is always 0 for datagpu devices
      metrics.m_dmaSize   = size;
      metrics.m_dmaBytes += size;
      // @todo: Is this the case here also?
      // dmaReadBulkIndex() returns a maximum size of m_pool.dmaSize(), never larger.
      // If the DMA overflowed the buffer, the excess is returned in a 2nd DMA buffer,
      // which thus won't have the expected header.  Take the exact match as an overflow indicator.
      if (size == m_pool.dmaSize()) {
        logging::critical("%d DMA overflowed buffer: %d vs %d", dmaIdx, size, m_pool.dmaSize());
        abort();
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
      ///buffer->index = dmaIdx;                    //   "
      ///event->mask |= (1 << lane);

      if (dsc->error) {
        logging::error("DMA error 0x%x",dsc->error);
        ///m_batchLast += 1;               // Account for the missing entry
        ///handleBrokenEvent(*event);
        freeDma(dmaIdx);                 // Leaves event mask = 0
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
        ///m_batchLast += 1;               // Account for the missing entry
        ///handleBrokenEvent(*event);
        freeDma(dmaIdx);                 // Leaves event mask = 0
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
          ///m_batchLast += 1;           // Account for the missing entry
          ///handleBrokenEvent(*event);
          freeDma(dmaIdx);             // Leaves event mask = 0
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

      // Allocate a pebble buffer
      ///auto counter       = m_pool.allocate(); // This can block
      /////auto pebbleIndex   = counter & (m_pool.nbuffers() - 1);
      ///auto pebbleIndex   = (lastPblIndex + dmaIdx) & (m_pool.nbuffers() - 1);
      ///event->pebbleIndex = pebbleIndex;
      ///
      ///if (pebbleIndex != ((lastPblIndex + dmaIdx) & (m_pool.nbuffers() - 1))) {
      ///  printf("*** pblIdx %u, last %u, Worker %d, dmaIdx %d, nbufs %d, pid %014lx, %014lx\n",
      ///         pebbleIndex, (lastPblIndex + dmaIdx) & (m_pool.nbuffers() - 1), m_worker, dmaIdx,
      ///         m_pool.nbuffers(), pid, timingHeader->pulseId());
      ///}
      ///lastPblIndex += 4;

      if (transitionId != XtcData::TransitionId::L1Accept) {
        if (transitionId != XtcData::TransitionId::SlowUpdate) {
          logging::info("GpuWorker  saw %s @ %u.%09u (%014lx)",
                        XtcData::TransitionId::name(transitionId),
                        timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                        timingHeader->pulseId());
          ///flushBatch = true;
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

      //if (pgpIndex != (m_batchStart + m_batchLast) & bufferMask)
      //  logging::error("%d Event counter mismatch: got %u, expected %u\n",
      //                 dmaIdx, pgpIndex, (m_batchStart + m_batchLast) & bufferMask);
      //if (evtCounter != m_lastEvtCtr + pgpIndex + 1) {
      //  logging::error("%d Event counter mismatch: got %u, expected %u\n",
      //                 dmaIdx, evtCounter, m_lastEvtCtr + pgpIndex + 1);
      //  ++m_nPgpJumps;
      //}

      // Make a new dgram in the pebble
      // It must be an EbDgram in order to be able to send it to the MEB
      ///if (!m_pool.pebble[pebbleIndex]) printf("*** pbl[%d] is NULL for ctr %d\n", pebbleIndex, pgpIndex);
      ///auto dgram = new(m_pool.pebble[pebbleIndex]) EbDgram(*th, det.nodeId, m_para.rogMask);
      ///if (!dgram) printf("*** dgram %p at pi %d for ctr %d\n", dgram, pebbleIndex, pgpIndex);

      metrics.m_nevents += 1;
      ///m_batchLast += 1;
      ///m_batchSize += nDmaRet;
      ///flushBatch |= m_batchSize == 4;           // @todo: arbitrary
      ///
      /////printf("*** dma %d nDmaRet %d, size %u, flushBatch %d\n",
      /////       dmaIdx, nDmaRet, m_batchSize.load(), flushBatch);
      ///
      ///if (flushBatch) {
      ///  // Ensure PGPReader::handle() doesn't complain about jumps
      ///  m_lastEvtCtr = evtCounter;
      ///
      ///  // Queue the batch to the Collector
      ///  logging::debug("Worker %d, idx %d pushing batch %u, size %zu\n",
      ///                 m_worker, dmaIdx, m_batchStart.load(), m_batchSize.load());
      ///  workerQueue.push({m_batchStart, m_batchSize});
      ///
      ///  // Reset to the beginning of the batch
      ///  m_batchLast  = 0;
      ///  m_batchStart = pgpIndex + 1;
      ///  m_batchSize  = 0;
      ///  flushBatch   = false;
      ///}
      m_dmaQueue.push(dmaIdx);
    }
  }

  m_dmaQueue.shutdown();

  logging::info("Shutting down GPU streams");
  m_terminate_d->store(1, cuda::memory_order_release);

  logging::debug("GpuWorker[%d] exiting\n", m_worker);
}
