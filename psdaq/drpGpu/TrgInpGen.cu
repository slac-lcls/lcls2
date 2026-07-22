#include "TrgInpGen.hh"

#include "Reader.hh"

#include "psalg/utils/SysLog.hh"
#include "psdaq/service/MetricExporter.hh"
#include "psdaq/service/EbDgram.hh"     // For TimingHeader
#include "psdaq/trigger/src/TriggerPrimitive.hh"
#include "psdaq/eb/src/eb.hh"

#include <sys/prctl.h>

#if 0                                   // Set to 1 to enable device metrics.  These affect performance.
#define DBG(stmt) stmt
#else
#define DBG(stmt)
#endif

using namespace XtcData;
using namespace Pds;
using namespace Drp;
using namespace Drp::Gpu;
using logging = psalg::SysLog;
using us_t = std::chrono::microseconds;

static const char* const RED_ON  = "\033[0;31m";
static const char* const RED_OFF = "\033[0m";
static const unsigned EvtCtrMask = 0xffffff;

struct tig_domain{ static constexpr char const* name{"TrgInpGen"}; };
using tig_scoped_range = nvtx3::scoped_range_in<tig_domain>;


TrgInpGen::TrgInpGen(const Parameters&                  para,
                     MemPoolGpu&                        pool,
                     const std::shared_ptr<Reader>&     reader,
                     Trg::TriggerPrimitive*             triggerPrimitive,
                     cudaExecutionContext_t             green_ctx,
                     const std::atomic<bool>&           terminate,
                     const cuda::std::atomic<unsigned>& terminate_d) :
  m_pool            (pool),
  m_triggerPrimitive(triggerPrimitive),
  m_terminate       (terminate),
  m_terminate_d     (terminate_d),
  m_retCode_d       (nullptr),
  m_reader          (reader),
  m_lastPid         (0),
  m_latPid          (0),
  m_lastComplete    (0),
  m_lastTid         (TransitionId::Unconfigure),
  m_lastData        {0, 0, 0, 0, 0, 0},
  m_para            (para)
{
  // Gather up the device readerQueuers in a device array
  auto& readerQueues = m_reader->readerQueues();
  chkError(cudaMalloc(&m_readerQueues_d, m_reader->nReaders() * sizeof(*m_readerQueues_d)));
  for (unsigned i = 0; i < m_reader->nReaders(); ++i) {
    chkError(cudaMemcpy(&m_readerQueues_d[i], &readerQueues[i].d, sizeof(*m_readerQueues_d), cudaMemcpyDefault));
  }

  // Set up buffer index queue for TrgInpGen to Host comms
  m_trgInpGenQueue.h = new RingIndexDtoH(pool.nbuffers());
  chkError(cudaMalloc(&m_trgInpGenQueue.d,                     sizeof(*m_trgInpGenQueue.d)));
  chkError(cudaMemcpy( m_trgInpGenQueue.d, m_trgInpGenQueue.h, sizeof(*m_trgInpGenQueue.d), cudaMemcpyDefault));

  // Get the range of priorities available [ greatest_priority, lowest_priority ]
  int prioLo;
  int prioHi;
  chkError(cudaDeviceGetStreamPriorityRange(&prioLo, &prioHi));
  int prio{prioLo-1};
  logging::debug("TrgInpGen stream priority (range: LOW: %d to HIGH: %d): %d", prioLo, prioHi, prio);

  // Create the TrgInpGen EB stream with higher priority than the Reader
  //chkFatal(cudaStreamCreateWithPriority(&m_stream, cudaStreamNonBlocking, prio));
  chkFatal(cudaExecutionCtxStreamCreate(&m_stream, green_ctx, cudaStreamNonBlocking, prio));
  logging::debug("Done with creating TrgInpGen stream");

  // Keep track of the index of the TrgInpGen stream
  chkError(cudaMalloc(&m_index_d,    sizeof(*m_index_d)));
  chkError(cudaMemset( m_index_d, 0, sizeof(*m_index_d)));

  // Set up a state variable
  chkError(cudaMalloc(&m_state_d,    sizeof(*m_state_d)));
  chkError(cudaMemset( m_state_d, 0, sizeof(*m_state_d)));

  // Create a location to capture error conditions returned by the trigger primitive
  chkError(cudaMalloc(&m_retCode_d,    sizeof(*m_retCode_d)));
  chkError(cudaMemset( m_retCode_d, 0, sizeof(*m_retCode_d)));

  // Prepare a metric for tracking kernel state
  chkError(cudaHostAlloc(&m_metrics.state, sizeof(*m_metrics.state), cudaHostAllocDefault));
  *m_metrics.state = 0;

  // Prepare counter metrics for tracking execution progress
  chkError(cudaHostAlloc(&m_metrics.rcvWtCtr, sizeof(*m_metrics.rcvWtCtr), cudaHostAllocDefault));
  *m_metrics.rcvWtCtr = 0;
  chkError(cudaHostAlloc(&m_metrics.fwdWtCtr, sizeof(*m_metrics.fwdWtCtr), cudaHostAllocDefault));
  *m_metrics.fwdWtCtr = 0;
}

TrgInpGen::~TrgInpGen()
{
  chkError(cudaGraphExecDestroy(m_graphExec));

  if (m_metrics.rcvWtCtr) {
    chkError(cudaFreeHost(m_metrics.rcvWtCtr));
    m_metrics.rcvWtCtr = nullptr;
  }
  if (m_metrics.fwdWtCtr) {
    chkError(cudaFreeHost(m_metrics.fwdWtCtr));
    m_metrics.fwdWtCtr = nullptr;
  }

  if (m_metrics.state) {
    chkError(cudaFreeHost(m_metrics.state));
    m_metrics.state = nullptr;
  }

  if (m_retCode_d)  chkError(cudaFree(m_retCode_d));
  m_retCode_d = nullptr;

  if (m_state_d)  chkError(cudaFree(m_state_d));
  m_state_d = nullptr;

  if (m_index_d)  chkError(cudaFree(m_index_d));
  m_index_d = nullptr;

  chkError(cudaStreamDestroy(m_stream));

  if (m_trgInpGenQueue.d)  chkError(cudaFree(m_trgInpGenQueue.d));
  delete m_trgInpGenQueue.h;
  m_trgInpGenQueue.d = nullptr;
  m_trgInpGenQueue.h = nullptr;

  if (m_readerQueues_d)  chkError(cudaFree(m_readerQueues_d));
  m_readerQueues_d = nullptr;
}

int TrgInpGen::setupMetrics(const std::shared_ptr<MetricExporter> exporter,
                            std::map<std::string, std::string>&   labels)
{
  *m_metrics.state = 0;
  *m_metrics.rcvWtCtr = 0;
  *m_metrics.fwdWtCtr = 0;
  exporter->add("DRP_tigState", labels, MetricType::Gauge,   [&](){ return m_metrics.state    ? *m_metrics.state    : 0; });
  exporter->add("DRP_tigRcv",   labels, MetricType::Counter, [&](){ return m_metrics.rcvWtCtr ? *m_metrics.rcvWtCtr : 0; });
  exporter->add("DRP_tigFwd",   labels, MetricType::Counter, [&](){ return m_metrics.fwdWtCtr ? *m_metrics.fwdWtCtr : 0; });

  m_metrics.pndWtCtr = 0;
  m_metrics.pidWtCtr = 0;
  exporter->add("DRP_colRcv",   labels, MetricType::Counter, [&](){ return m_metrics.pndWtCtr; });
  exporter->add("DRP_pidWtCtr", labels, MetricType::Counter, [&](){ return m_metrics.pidWtCtr; });

  exporter->add("DRP_colQueOcc", labels, MetricType::Gauge,   [&](){ return m_trgInpGenQueue.h->occupancy(); });

  m_metrics.nEvents = 0L;
  exporter->add("DRP_evtCtr",   labels, MetricType::Counter, [&](){ return m_metrics.nEvents; });
  exporter->add("drp_event_rate", labels, MetricType::Rate,
                [&](){return m_metrics.nEvents;});

  m_metrics.nDmaRet = 0;
  exporter->add("drp_num_dma_ret", labels, MetricType::Gauge,
                [&](){return m_metrics.nDmaRet;});
  m_metrics.dmaBytes = 0;
  exporter->add("drp_pgp_byte_rate", labels, MetricType::Rate,
                [&](){return m_metrics.dmaBytes;});
  m_metrics.dmaSize = 0;
  exporter->add("drp_dma_size", labels, MetricType::Gauge,
                [&](){return m_metrics.dmaSize;});
  exporter->add("drp_th_latency", labels, MetricType::Gauge,
                [&](){return m_metrics.latency;});
  m_metrics.nDmaErrors = 0;
  exporter->add("drp_num_dma_errors", labels, MetricType::Gauge,
                [&](){return m_metrics.nDmaErrors;});
  m_metrics.nNoComRoG = 0;
  exporter->add("drp_num_no_common_rog", labels, MetricType::Gauge,
                [&](){return m_metrics.nNoComRoG;});
  m_metrics.nMissingRoGs = 0;
  exporter->add("drp_num_missing_rogs", labels, MetricType::Gauge,
                [&](){return m_metrics.nMissingRoGs;});
  m_metrics.nTmgHdrError = 0;
  exporter->add("drp_num_th_error", labels, MetricType::Gauge,
                [&](){return m_metrics.nTmgHdrError;});
  m_metrics.nPgpJumps = 0;
  exporter->add("drp_num_pgp_jump", labels, MetricType::Gauge,
                [&](){return m_metrics.nPgpJumps;});

  return 0;
}

int TrgInpGen::_setupGraph()
{
  logging::debug("Recording TrgInpGen graph");

  // Build the graph
  cudaGraph_t graph = _recordGraph(m_stream);
  if (graph == 0) {
    return -1;
  }

  // Instantiate the graph
  if (chkError(cudaGraphInstantiate(&m_graphExec, graph, cudaGraphInstantiateFlagDeviceLaunch),
               "TrgInpGen graph create failed")) {
    return -1;
  }

  // No need to hang on to the stream info
  cudaGraphDestroy(graph);

  // Upload the graph so it can be launched by the scheduler kernel later
  logging::debug("Uploading TrgInpGen graph...");
  if (chkError(cudaGraphUpload(m_graphExec, m_stream), "TrgInpGen graph upload failed")) {
    return -1;
  }

  logging::debug("Done recording TrgInpGen graph");
  return 0;
}

static __device__ unsigned lReader = 0;

// This kernel receives indices from the DMA stream
static __global__
void _trgInpGenRcv(unsigned*      const        __restrict__ state,
                   unsigned*      const        __restrict__ index,
                   RingIndexDtoD* const* const __restrict__ readerQueues, // [nReaders]
                   unsigned       const                     nReaders,
                   unsigned       const                     nRdrShft,     // log2(nReaders)
                   uint64_t*      const        __restrict__ stateMon,
                   uint64_t*      const        __restrict__ rcvWtCtr)
{
  //printf("### _trgInpGenRcv: state %u, index %u\n", *state, *index);

  if (*state == 0) {
    DBG(*stateMon = 2;)
    // Get the intermediate buffer index
    unsigned idx;
    unsigned ns{8};
    while (!readerQueues[lReader]->pop(&idx)) {
      __nanosleep(ns);
      if (ns < 256)  ns *= 2;
      else {
        DBG(*stateMon = 4;)
        return;
      }
    }
    idx = (idx << nRdrShft) | lReader;
    //printf("### _trgInpGenRcv: got idx %u from reader %u, hd %u, tl %u, occ %u\n", idx, lReader,
    //       readerQueues[lReader]->head(), readerQueues[lReader]->tail(), readerQueues[lReader]->occupancy());
    *state = 1;
    DBG(*stateMon = 5; ++(*rcvWtCtr);)

    // Advance index and to next reader
    *index = idx;
    atomicInc(&lReader, nReaders-1);
  }
}

// This will re-launch the current graph
static __global__
void _trgInpGenLoop(unsigned*      const  __restrict__ state,
                    unsigned*      const  __restrict__ index,
                    RingIndexDtoH* const  __restrict__ trgInpGenQueue,
                    uint64_t*      const  __restrict__ stateMon,
                    uint64_t*      const  __restrict__ fwdWtCtr,
                    cuda::std::atomic<unsigned> const& terminate)
{
  if (*state == 2) {
    DBG(*stateMon = 10;)
    // Push index to host and increment to the next one
    //printf("### _trgInpGenLoop: push index %u\n", *index);
    bool rc;
    unsigned ns{8};
    auto const idx{*index};
    while ( (rc = !trgInpGenQueue->push(idx)) ) {
      __nanosleep(ns);
      if (ns < 256)  ns *= 2;
      else {
        DBG(*stateMon = 11;)
        break;
      }
    }
    if (!rc) {
      //printf("### _trgInpGenLoop: pushed index %u\n", *index);
      *state = 0;
      DBG(*stateMon = 12; ++(*fwdWtCtr);)
    }
  }

  // This will re-launch the current graph
  //printf("### _trgInpGenLoop: relaunch\n");
  if (!terminate.load(cuda::std::memory_order_acquire))  {
    cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
  }
  else {
    //printf("### _trgInpGenLoop: Terminate is True: not relaunching\n");
    lReader = 0;                        // Reset for possible next time
  }
}

cudaGraph_t TrgInpGen::_recordGraph(cudaStream_t stream)
{
  tig_scoped_range r{/*"TrgInpGen::_recordGraph"*/}; // Expose function name via NVTX

  auto hostWrtBufs    = m_pool.hostWrtBufs();
  auto hostWrtBufsCnt = m_pool.hostWrtBufsSize() / sizeof(*hostWrtBufs);
  auto calibBuffers   = m_pool.calibBuffers_d();
  auto calibBufsCnt   = m_pool.calibBufsSize() / sizeof(*calibBuffers);
  auto nReaders       = m_reader->nReaders();
  auto const nRdrShft = ffs(nReaders) - 1; // log2(nReaders)

  if (chkError(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "TrgInpGen stream begin capture failed")) {
    return 0;
  }

  // Find which pebble buffers are ready for processing
  _trgInpGenRcv<<<1, 1, 0, stream>>>(m_state_d,
                                     m_index_d,
                                     m_readerQueues_d,
                                     nReaders,
                                     nRdrShft,
                                     m_metrics.state,
                                     m_metrics.rcvWtCtr);

  // Process calibBuffers[tail] into TEB input data placed at the end of hostWrtBufs[tail]
  if (m_triggerPrimitive) { // else this DRP doesn't provide TEB input
    m_triggerPrimitive->event(stream,
                              m_state_d,
                              calibBuffers,
                              calibBufsCnt,
                              hostWrtBufs,
                              hostWrtBufsCnt,
                              m_index_d,
                              m_retCode_d);
  }

  // Post the buffer to the host and relaunch
  _trgInpGenLoop<<<1, 1, 0, stream>>>(m_state_d,
                                      m_index_d,
                                      m_trgInpGenQueue.d,
                                      m_metrics.state,
                                      m_metrics.fwdWtCtr,
                                      m_terminate_d);

  cudaGraph_t graph;
  if (chkError(cudaStreamEndCapture(stream, &graph),
               "TrgInpGen stream end capture failed")) {
    return 0;
  }

  return graph;
}

bool TrgInpGen::setup()                 // Called during phase 1 of Configure
{
  // Prepare the TrgInpGen graph
  if (_setupGraph()) {
    logging::error("Failed to set up TrgInpGen graph");
    return true;
  }
  return false;
}

bool TrgInpGen::startup()               // Called during phase 1 of Configure
{
  logging::info("TrgInpGen starting");

  resetEventCounter();

  // Launch the TrgInpGen graph
  logging::debug("Launching TrgInpGen graph");
  chkFatal(cudaGraphLaunch(m_graphExec, m_stream));

  return false;
}

void TrgInpGen::start(SPSCQueue<unsigned>& collectorQueue)
{
  m_receiverThread = std::thread(&TrgInpGen::_receiver, this, std::ref(collectorQueue));
}

void TrgInpGen::shutdown()
{
  // Wait for the TrgInpGen receiver thread to finish
  if (m_receiverThread.joinable()) {
    m_receiverThread.join();
    logging::info("TrgInpGen receiver thread finished");
  }
}

void TrgInpGen::_receiver(SPSCQueue<unsigned>& collectorQueue)
{
  tig_scoped_range r{/*"TrgInpGen::receiver"*/}; // Expose function name via NVTX

  logging::info("TrgInpGen receiver is starting with process ID %lu\n", syscall(SYS_gettid));
  if (prctl(PR_SET_NAME, "drp_gpu/TigRcvr", 0, 0, 0) == -1) {
    perror("prctl");
  }

  // If triggers had been left running, they will have been stopped during Allocate
  // Flush anything that accumulated
  m_reader->flush();

  const auto     hostWrtBufs    = m_pool.hostWrtBufs(); // When no error, hdrs in all are the same
  const auto     hostWrtBufsCnt = m_pool.hostWrtBufsSize() / sizeof(*hostWrtBufs);
  const uint32_t bufferMask     = m_pool.nbuffers() - 1;

  while (!m_terminate.load(std::memory_order_relaxed)) {
    unsigned nRet = 0;
    unsigned index = 0;
    while (m_trgInpGenQueue.h->pop(&index)) {
      tig_scoped_range loop_range{/*"TrgInpGen::receive", */nvtx3::payload{index}};
      ++m_metrics.pndWtCtr;

      const auto dmaDsc       = (DmaDsc*)&hostWrtBufs[index * hostWrtBufsCnt];
      const auto timingHeader = (TimingHeader*)&dmaDsc[1];

      if (m_para.verbose > 2) {
        printf("*** TrgInpGen::receive: dmaDsc[%u] %p, th %p\n", index, dmaDsc, timingHeader);
        const auto& p = (const uint32_t*)dmaDsc;
        printf("*** TrgInpGen::receive: dmaBuf %08x %08x | %08x %08x %08x %08x %08x %08x\n",
               p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
      }

      uint64_t pid = timingHeader->pulseId();
      if (pid <= m_lastPid) [[unlikely]]
        logging::error("%s: PulseId did not advance: %014lx <= %014lx", __PRETTY_FUNCTION__, pid, m_lastPid);
      m_lastPid = pid;

      // Check for DMA buffer overflow
      if (dmaDsc->overflow()) [[unlikely]] {
        logging::critical("DMA overflowed buffer: size %d B vs %d B", dmaDsc->size, m_pool.dmaSize());
        abort();
      }

      // Measure TimingHeader arrival latency as early as possible
      if (pid - m_latPid > 1300000/14) {  // 10 Hz
        m_metrics.latency = Eb::latency<us_t>(timingHeader->time);
        m_latPid = pid;
      }

#ifdef HOST_REARMS_DMA
      // Write to the DMA start register in the FPGA
      unsigned dmaIdx = index % m_pool.dmaCount();
      //printf("*** TrgInpGen::receive: Enable write to DMA buffer %u\n", dmaIdx);
      auto rc = gpuSetWriteEn(m_pool.fd(), dmaIdx);
      if (rc < 0) [[unlikely]] {
        logging::critical("Failed to reenable buffer %u for write: %zd: %m", dmaIdx, rc);
        abort();
      }
#endif

      logging::debug("dma %u, hdr %08x, sz %08x = %u", index, dmaDsc->header, dmaDsc->size, dmaDsc->size);
      logging::debug("idx %u  th: ctl %02x, pid %014lx, ts %016lx, env %08x, ctr %08x, opq %08x %08x",
                     index, timingHeader->control(), timingHeader->pulseId(), timingHeader->time.value(),
                     timingHeader->env, timingHeader->evtCounter,
                     timingHeader->_opaque[0], timingHeader->_opaque[1]);

      uint32_t size = dmaDsc->size;         // Size of the DMA
      m_metrics.dmaSize   = size;
      m_metrics.dmaBytes += size;

      if (timingHeader->error()) [[unlikely]] {
        if (m_metrics.nTmgHdrError++ < 5) { // Limit prints at rate
          logging::error("Timing header error bit is set");
        }
      }

      uint32_t evtCounter = timingHeader->evtCounter & EvtCtrMask;
      unsigned pgpIndex = evtCounter & bufferMask;
      PGPEvent* event = &m_pool.pgpEvents[pgpIndex];
      constexpr uint32_t lane{0}; // The lane is always 0 for GPU-enabled PGP devices
      DmaBuffer* buffer = &event->buffers[lane];
      buffer->size = size;
      buffer->index = index;                // Intermediate buffer index; NOT DMA buffer index
      if (event->mask) [[unlikely]] {
        logging::critical("PGPEvent mask (%02x) != 0 for pgpIdx %u, ctr %u\n",
                          event->mask, pgpIndex, evtCounter);
        abort();
      }
      if (((1 << lane) & m_para.laneMask) == 0) [[unlikely]] {
        logging::error("Lane %u is not in laneMask 0x%02", lane, m_para.laneMask);
      }
      event->mask |= (1 << lane);

      m_pool.allocateDma(); // DMA buffer was allocated when f/w incremented evtCounter

      // Check whether the DMA is reporting an error
      // @todo: C1100: if (dmaDsc->header ^ ~dmaDsc->errorMask()) [[unlikely]] {
      if (dmaDsc->header & dmaDsc->errorMask()) [[unlikely]] {    // Ignore SOF for KCU usage
        // Assume we can recover from non-overflow DMA errors
        if (m_metrics.nDmaErrors++ < 5) {   // Limit prints at rate
          logging::error("DMA error 0x%08x", dmaDsc->header);
        }
        // This assumes the DMA succeeded well enough that evtCounter is valid
        handleBrokenEvent(*event);
        m_reader->freeDma(event);       // Leaves event mask = 0
        break;
      }

      XtcData::TransitionId::Value transitionId = timingHeader->service();
      const uint32_t* data = reinterpret_cast<const uint32_t*>(timingHeader);
      logging::debug("PGPReader  size %u  hdr %016lx.%016lx.%08x  dma hdr 0x%08x",
                     size,
                     reinterpret_cast<const uint64_t*>(data)[0], // PulseId
                     reinterpret_cast<const uint64_t*>(data)[1], // Timestamp
                     reinterpret_cast<const uint32_t*>(data)[4], // env
                     dmaDsc->header);

      if (transitionId == TransitionId::BeginRun) {
        resetEventCounter();            // Compensate for the ClearReadout sent before BeginRun
      }
      if (evtCounter != ((m_lastComplete + 1) & EvtCtrMask)) [[unlikely]] {
        if (m_lastTid != TransitionId::Unconfigure) {
          if ((m_metrics.nPgpJumps++ < 5) || m_para.verbose) { // Limit prints at rate
            auto evtCntDiff = evtCounter - m_lastComplete;
            logging::error("%sPGPReader: Jump in TimingHeader evtCounter %u -> %u | difference %d, DMA size %u%s, index %u",
                           RED_ON, m_lastComplete, evtCounter, evtCntDiff, size, RED_OFF, index);
            logging::error("new data: %08x %08x %08x %08x %08x %08x  (%s)",
                           data[0], data[1], data[2], data[3], data[4], data[5], TransitionId::name(transitionId));
            logging::error("lastData: %08x %08x %08x %08x %08x %08x  (%s)",
                           m_lastData[0], m_lastData[1], m_lastData[2], m_lastData[3], m_lastData[4], m_lastData[5], TransitionId::name(m_lastTid));
          }
          handleBrokenEvent(*event);
          m_reader->freeDma(event);     // Leaves event mask = 0
          abort();
          break;                        // Throw away out-of-sequence events
        } else if (transitionId != TransitionId::Configure) {
          m_reader->freeDma(event);     // Leaves event mask = 0
          break;                        // Drain
        }
      }
      m_lastComplete = evtCounter;
      m_lastTid = transitionId;
      memcpy(m_lastData, data, 24);

      auto rogs = timingHeader->readoutGroups();
      if ((rogs & (1 << m_para.partition)) == 0) {
        logging::debug("%s @ %u.%09u (%014lx) without common readout group (%u) in env 0x%08x",
                       XtcData::TransitionId::name(transitionId),
                       timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                       timingHeader->pulseId(), m_para.partition, timingHeader->env);
        ++m_lastComplete;
        handleBrokenEvent(*event);
        m_reader->freeDma(event);       // Leaves event mask = 0
        ++m_metrics.nNoComRoG;
        break;
      }
      if (transitionId == XtcData::TransitionId::SlowUpdate) {
        uint16_t missingRogs = m_para.rogMask & ~rogs;
        if (missingRogs) [[unlikely]] {
          logging::debug("%s @ %u.%09u (%014lx) missing readout group(s) (0x%04x) in env 0x%08x",
                         XtcData::TransitionId::name(transitionId),
                         timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                         timingHeader->pulseId(), missingRogs, timingHeader->env);
          ++m_lastComplete;
          handleBrokenEvent(*event);
          m_reader->freeDma(event);     // Leaves event mask = 0
          ++m_metrics.nMissingRoGs;
          break;
        }
      }

      if (transitionId != XtcData::TransitionId::L1Accept) {
        if (transitionId != XtcData::TransitionId::SlowUpdate) {
          logging::info("PGPReader  saw %12s @ %u.%09u (%014lx)",
                        XtcData::TransitionId::name(transitionId),
                        timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                        timingHeader->pulseId());
        }
        else {
          logging::debug("PGPReader  saw %12s @ %u.%09u (%014lx)",
                         XtcData::TransitionId::name(transitionId),
                         timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                         timingHeader->pulseId());
        }
      }

      ++nRet;
      ++m_metrics.nEvents;
    }

    m_metrics.nDmaRet = nRet;
    if (nRet)  collectorQueue.push(nRet);
  }

  collectorQueue.shutdown();

  logging::info("TrgInpGen receiver is exiting");
}
