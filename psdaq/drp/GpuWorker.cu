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

//https://github.com/slaclab/surf/blob/main/axi/dma/rtl/v2/AxiStreamDmaV2Write.vhd
struct DmaDsc
{
  uint32_t error;
  uint32_t size;
  uint32_t _rsvd[6];
};

};


MemPoolGpu::MemPoolGpu(Parameters& para) :
  MemPool(para)
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
    printf("*** Worker %d, %s unit %d\n", worker, para.device.substr(0, pos).c_str(), unit);
    auto& seg = m_segs[worker];

    std::string device(para.device.substr(0, pos+1) + std::to_string(unit));
    seg.fd = open(device.c_str(), O_RDWR);
    if (seg.fd < 0) {
      logging::critical("Error opening %s: %m", device.c_str());
      abort();
    }
    logging::info("PGP device '%s' opened", device.c_str());

    // @todo: Get the DMA size from somewhere
    m_dmaSize = 128*1024;

    // Clear out any left-overs from last time
    int res = gpuRemNvidiaMemory(seg.fd);
    if (res < 0)  logging::error("Error in gpuRemNvidiaMemory\n");
    logging::debug("Done with gpuRemNvidiaMemory() cleanup\n");

    // Allocate buffers on the GPU
    // This handles allocating buffers on the device and registering them with the driver.
    auto size = dmaSize();
    for (unsigned i = 0; i < dmaCount(); ++i) {
      if (_gpuMapFpgaMem(seg.fd, seg.dmaBuffers[i], 0, size, 1) != 0) {
        logging::critical("Worker %d failed to alloc buffer list at number %zd", worker, i);
        abort();
      }
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
      _gpuUnmapFpgaMem(seg.dmaBuffers[i]);
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
    if (m_setMaskBytesDone) {
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
                m_setMaskBytesDone = true;
            }
        }
    }
    return retval;
}

int MemPoolGpu::_gpuMapFpgaMem(int fd, CUdeviceptr& buffer, uint64_t offset, size_t size, int write)
{
  if (chkError(cuMemAlloc(&buffer, size))) {
    return -1;
  }
  cuMemsetD8(buffer, 0, size);

  int flag = 1;
  // This attribute is required for peer shared memory. It will synchronize every synchronous memory operation on this block of memory.
  if (chkError(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, buffer))) {
    cuMemFree(buffer);
    return -1;
  }

  if (gpuAddNvidiaMemory(fd, write, buffer, size) < 0) {
    logging::error("gpuAddNvidiaMemory failed");
    cuMemFree(buffer);
    return -1;
  }

  return 0;
}

void MemPoolGpu::_gpuUnmapFpgaMem(CUdeviceptr& buffer)
{
  chkError(cuMemFree(buffer));
}


GpuWorker::GpuWorker(unsigned worker, const Parameters& para, MemPoolGpu& pool) :
  m_pool        (pool),
  m_batchStart  (1),
  m_batchSize   (0),
  m_dmaIndex    (0),
  m_worker      (worker),
  m_para        (para),
  m_nDmaErrors  (0),
  m_nNoComRoG   (0),
  m_nMissingRoGs(0),
  m_nTmgHdrError(0)
{
  ////////////////////////////////////
  // Allocate streams
  ////////////////////////////////////

  /** Allocate a stream per buffer **/
  m_streams.resize(m_pool.dmaCount());
  for (auto& stream : m_streams) {
    chkFatal(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "Error creating streams");
  }
  logging::debug("Done with creating streams\n");
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

void GpuWorker::start(SPSCQueue<Batch>& workerQueue, Detector* det)
{
  // Launch one thread per stream
  for (unsigned i = 0; i < m_streams.size(); ++i) {
    m_threads.emplace_back(&GpuWorker::_reader, std::ref(*this),
                           i, std::ref(workerQueue), std::ref(*det));
  }
}

void GpuWorker::stop()
{
  logging::warning("GpuWorker::stop called for worker %d\n", m_worker);

  chkError(cuCtxSetCurrent(m_pool.context().context()));
  printf("*** context set\n");
  //chkError(cuCtxSynchronize());
  //printf("*** context synced\n");

  // Stop and clean up the threads
  auto& seg = m_pool.segs()[m_worker];
  for (unsigned i = 0; i < m_threads.size(); ++i) {
    // @todo: Need to trigger the streams here
    printf("*** %d Set handshake\n", i);
    chkFatal(cuStreamWriteValue32(m_streams[i], seg.dmaBuffers[i] + 4, 0x01, 0));
    //printf("*** %d handshake sync\n", i);
    //chkError(cuCtxSynchronize());
    //printf("*** %d context synced\n", i);
    //chkError(cuStreamSynchronize(m_streams[i]));
    printf("*** Worker %d, thread %d handshake done\n", m_worker, i);
  }
  for (unsigned i = 0; i < m_threads.size(); ++i) {
    printf("*** Worker %d, thread %d join\n", m_worker, i);
    m_threads[i].join();
    printf("*** Worker %d, thread %d joined\n", m_worker, i);
  }
}

void GpuWorker::freeDma(PGPEvent* event)
{
  event->mask = 0;

  m_pool.freeDma(1, nullptr);
}

void GpuWorker::handleBrokenEvent(const PGPEvent&)
{
  m_batchSize += 1; // Broken events must be included in the batch since f/w advanced evtCounter
}

void GpuWorker::resetEventCounter()
{
  m_batchStart = 1;
  m_batchSize = 0;
  m_batchLast = 0;
}

// @todo: Do we still want these?
//void GpuWorker::timingHeaders(unsigned index, TimingHeader* buffer)
//{
//  auto idx = index & (m_streams.size() - 1);
//  chkFatal(cuMemcpyDtoH((void*)buffer, m_pool.dmaBuffers[idx], sizeof(*buffer)));
//}
//
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

void GpuWorker::_reader(unsigned dmaIdx, SPSCQueue<Batch>& workerQueue, Detector& det)
{
  logging::info("GpuWorker[%d].reader[%d] starting\n", m_worker, dmaIdx);
  chkError(cuCtxSetCurrent(m_pool.context().context()));  // Needed, else kernels misbehave

  dmaTarget(DmaTgt_t::GPU); // Ensure that timing messages are DMAed to the GPU

  resetEventCounter();

  const uint32_t bufferMask = m_pool.nbuffers() - 1;

  size_t    size         = sizeof(DmaDsc)+sizeof(TimingHeader); //m_pool.dmaSize();
  uint32_t* hostWriteBuf = (uint32_t*)malloc(size);

  // Handle L1Accepts, SlowUpdates and Disable
  bool        flushBatch = false;
  const auto& stream     = m_streams[dmaIdx];
  const auto& seg        = m_pool.segs()[m_worker];
  auto        dmaBuffer  = seg.dmaBuffers[dmaIdx];
  auto        fd         = seg.fd;
  while (true) {

    // Clear the GPU memory handshake space to zero
    //logging::debug("%d clear memory\n", dmaIdx);
    chkFatal(cuStreamWriteValue32(stream, dmaBuffer + 4, 0x00, 0));

    // Write to the DMA start register in the FPGA
    //logging::debug("Trigger write to buffer %d\n", dmaIdx);
    chkError(cuStreamSynchronize(stream));
    auto rc = gpuSetWriteEn(fd, dmaIdx);
    if (rc < 0) {
      logging::critical("Failed to reenable buffer %d for write: %zd\n", dmaIdx, rc);
      perror("gpuSetWriteEn");
      abort();
    }

    // Spin on the handshake location until the value is greater than or equal to 1
    // This waits for the data to arrive in the GPU before starting the processing
    logging::debug("%d Wait memory value\n", dmaIdx);
    chkFatal(cuStreamWaitValue32(stream, dmaBuffer + 4, 0x1, CU_STREAM_WAIT_VALUE_GEQ));
    chkError(cuStreamSynchronize(stream));
    logging::debug("%d Done waiting\n", dmaIdx);
    unsigned nDmaRet = 1;
    m_pool.countDma();

    chkError(cuMemcpyDtoHAsync(hostWriteBuf, dmaBuffer, sizeof(DmaDsc)+sizeof(TimingHeader), stream));
    cuStreamSynchronize(stream);
    logging::debug("%d DtoH done\n", dmaIdx);

    const auto dsc = (DmaDsc*)hostWriteBuf;
    logging::debug("*** dma %d hdr: err %08x,  sz %08x, rsvd %08x %08x %08x %08x %08x %08x\n",
                   dmaIdx, dsc->error, dsc->size, dsc->_rsvd[0], dsc->_rsvd[1], dsc->_rsvd[2],
                   dsc->_rsvd[3], dsc->_rsvd[4], dsc->_rsvd[5]);
    const auto th  = (TimingHeader*)&hostWriteBuf[8];
    logging::debug("**G dma %d  th: ctl %02x, pid %014lx, ts %016lx, env %08x, ctr %08x, opq %08x %08x\n",
                   dmaIdx, th->control(), th->pulseId(), th->time.value(), th->env, th->evtCounter,
                   th->_opaque[0], th->_opaque[1]);

    uint32_t size = dsc->size;
    uint32_t lane = 0;                  // The lane is always 0 for datagpu devices
    m_dmaSize   = size;
    m_dmaBytes += size;
    // @todo: Is this the case here also?
    // dmaReadBulkIndex() returns a maximum size of m_pool.dmaSize(), never larger.
    // If the DMA overflowed the buffer, the excess is returned in a 2nd DMA buffer,
    // which thus won't have the expected header.  Take the exact match as an overflow indicator.
    if (size == m_pool.dmaSize()) {
      logging::critical("%d DMA overflowed buffer: %d vs %d", dmaIdx, size, m_pool.dmaSize());
      abort();
    }

    const Pds::TimingHeader* timingHeader = th; //det.getTimingHeader(index);
    uint32_t evtCounter = timingHeader->evtCounter & 0xffffff;
    unsigned pgpIndex = evtCounter & bufferMask;
    PGPEvent*  event  = &m_pool.pgpEvents[pgpIndex];
    DmaBuffer* buffer = &event->buffers[lane]; // @todo: Do we care about this?
    buffer->size = size;                       //   "
    buffer->index = dmaIdx;                    //   "
    event->mask |= (1 << lane);

    if (dsc->error) {
        logging::error("DMA with error 0x%x",dsc->error);
        m_batchLast += 1;               // Account for the missing entry
        handleBrokenEvent(*event);
        freeDma(event);                 // Leaves event mask = 0
        ++m_nDmaErrors;
        continue;
    }

    if (timingHeader->error()) {
        logging::error("Timing header error bit is set");
        ++m_nTmgHdrError;
    }
    XtcData::TransitionId::Value transitionId = timingHeader->service();

    auto rogs = timingHeader->readoutGroups();
    if ((rogs & (1 << m_para.partition)) == 0) {
        logging::debug("%s @ %u.%09u (%014lx) without common readout group (%u) in env 0x%08x",
                       XtcData::TransitionId::name(transitionId),
                       timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                       timingHeader->pulseId(), m_para.partition, timingHeader->env);
        m_batchLast += 1;               // Account for the missing entry
        handleBrokenEvent(*event);
        freeDma(event);                 // Leaves event mask = 0
        ++m_nNoComRoG;
        continue;
    }

    // @todo: Shouldn't this chack for missing RoGs on all transitions?
    if (transitionId == XtcData::TransitionId::SlowUpdate) {
        uint16_t missingRogs = m_para.rogMask & ~rogs;
        if (missingRogs) {
            logging::debug("%s @ %u.%09u (%014lx) missing readout group(s) (0x%04x) in env 0x%08x",
                           XtcData::TransitionId::name(transitionId),
                           timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                           timingHeader->pulseId(), missingRogs, timingHeader->env);
            m_batchLast += 1;           // Account for the missing entry
            handleBrokenEvent(*event);
            freeDma(event);             // Leaves event mask = 0
            ++m_nMissingRoGs;
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
    auto counter       = m_pool.allocate(); // This can block
    auto pebbleIndex   = counter & (m_pool.nbuffers() - 1);
    event->pebbleIndex = pebbleIndex;

    if (transitionId != XtcData::TransitionId::L1Accept) {
      if (transitionId != XtcData::TransitionId::SlowUpdate) {
        logging::info("GpuWorker  saw %s @ %u.%09u (%014lx)",
                      XtcData::TransitionId::name(transitionId),
                      timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                      timingHeader->pulseId());
        flushBatch = true;
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

    if (pgpIndex != m_batchStart + m_batchLast)
      logging::error("%d Event counter mismatch: got %u, expected %u\n",
                     dmaIdx, pgpIndex, m_batchStart + m_batchLast);

    // Make a new dgram in the pebble
    // It must be an EbDgram in order to be able to send it to the MEB
    auto dgram = new(m_pool.pebble[pebbleIndex]) EbDgram(*th, det.nodeId, m_para.rogMask);

    // @todo: Process the data to extract TEB input and calibrate.  Also reduce/compress?
    //if (timingHeader->service() == TransitionId::L1Accept)
    //  det.event(*th, event);
    //else if (timingHeader->service() == TransitionId::SlowUpdate)
    //  det.slowUpdate(*th);

    m_batchLast += 1;
    m_batchSize += nDmaRet;
    flushBatch |= m_batchSize == 4;           // @todo: arbitrary

    //logging::debug("*** %d nDmaRet %d, size %u, full %d, flushBatch %d\n",
    //               dmaIdx, nDmaRet, m_batchSize.load(), full, flushBatch);

    if (flushBatch) {
      // Ensure PGPReader::handle() doesn't complain about jumps
      m_lastEvtCtr = evtCounter;

      // Queue the batch to the Collector
      logging::debug("Worker %d, idx %d pushing batch %u, size %zu\n",
                     m_worker, dmaIdx, m_batchStart.load(), m_batchSize.load());
      workerQueue.push({m_batchStart, m_batchSize});

      // Reset to the beginning of the batch
      m_batchLast  = 0;
      m_batchStart = pgpIndex + 1;
      m_batchSize  = 0;
      flushBatch   = false;
    }
  }

  // Clean up
  free(hostWriteBuf);

  logging::debug("GpuWorker[%d].reader[%d] exiting\n", m_worker, dmaIdx);
}
