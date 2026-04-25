#include "Collector.hh"

#include "Detector.hh"
#include "Reader.hh"

#include "psalg/utils/SysLog.hh"
#include "psdaq/service/MetricExporter.hh"
#include "psdaq/service/EbDgram.hh"     // For TimingHeader
#include "psdaq/trigger/TriggerPrimitive.hh"
#include "psdaq/eb/eb.hh"

// Uncomment to dump a tracebuffer when a PGPReader event counter jump occurs
//#define USE_TRACEBUFFER

using namespace XtcData;
using namespace Pds;
using namespace Drp;
using namespace Drp::Gpu;
using logging = psalg::SysLog;
using us_t = std::chrono::microseconds;

static const char* const RED_ON  = "\033[0;31m";
static const char* const RED_OFF = "\033[0m";
static const unsigned EvtCtrMask = 0xffffff;

struct col_domain{ static constexpr char const* name{"Collector"}; };
using col_scoped_range = nvtx3::scoped_range_in<col_domain>;


Collector::Collector(const Parameters&                  para,
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
  m_pebbleQueue     (reader->pebbleQueue()), // Pebble buffer index queue
  m_readerQueue     (reader->readerQueue()), // Buffer index queue for Reader to Collector comms
  m_lastPid         (0),
  m_latPid          (0),
  m_lastComplete    (0),
  m_lastTid         (TransitionId::Unconfigure),
  m_para            (para)
{
  // Set up buffer index queue for Collector to Host comms
  m_collectorQueue.h = new RingIndexDtoH(pool.nbuffers());
  chkError(cudaMalloc(&m_collectorQueue.d,                     sizeof(*m_collectorQueue.d)));
  chkError(cudaMemcpy( m_collectorQueue.d, m_collectorQueue.h, sizeof(*m_collectorQueue.d), cudaMemcpyHostToDevice));

  // Get the range of priorities available [ greatest_priority, lowest_priority ]
  int prioLo;
  int prioHi;
  chkError(cudaDeviceGetStreamPriorityRange(&prioLo, &prioHi));
  int prio{prioLo-1};
  logging::debug("Collector stream priority (range: LOW: %d to HIGH: %d): %d", prioLo, prioHi, prio);

  // Create the Collector EB stream with higher priority than the Reader
  //chkFatal(cudaStreamCreateWithPriority(&m_stream, cudaStreamNonBlocking, prio));
  chkFatal(cudaExecutionCtxStreamCreate(&m_stream, green_ctx, cudaStreamNonBlocking, prio));
  logging::debug("Done with creating collector stream");

  // Keep track of the index of the Collector stream
  chkError(cudaMalloc(&m_index_d,    sizeof(*m_index_d)));
  chkError(cudaMemset( m_index_d, 0, sizeof(*m_index_d)));

  // Set up a state variable
  chkError(cudaMalloc(&m_state_d,    sizeof(*m_state_d)));
  chkError(cudaMemset( m_state_d, 0, sizeof(*m_state_d)));

  // Create a location to capture error conditions returned by the trigger primitive
  chkError(cudaMalloc(&m_retCode_d,    sizeof(*m_retCode_d)));
  chkError(cudaMemset( m_retCode_d, 0, sizeof(*m_retCode_d)));

  // Prepare a metric for tracking kernel state
  chkError(cudaHostAlloc(&m_metrics.state.h, sizeof(*m_metrics.state.h), cudaHostAllocDefault));
  chkError(cudaHostGetDevicePointer(&m_metrics.state.d, m_metrics.state.h, 0));
  *m_metrics.state.h = 0;

  // Prepare counter metrics for tracking execution progress
  chkError(cudaHostAlloc(&m_metrics.rcvWtCtr.h, sizeof(*m_metrics.rcvWtCtr.h), cudaHostAllocDefault));
  chkError(cudaHostGetDevicePointer(&m_metrics.rcvWtCtr.d, m_metrics.rcvWtCtr.h, 0));
  *m_metrics.rcvWtCtr.h = 0;
  chkError(cudaHostAlloc(&m_metrics.fwdWtCtr.h, sizeof(*m_metrics.fwdWtCtr.h), cudaHostAllocDefault));
  chkError(cudaHostGetDevicePointer(&m_metrics.fwdWtCtr.d, m_metrics.fwdWtCtr.h, 0));
  *m_metrics.fwdWtCtr.h = 0;

  // Prepare the Collector graph
  if (_setupGraph()) {
    logging::critical("Failed to set up Collector graph");
    abort();
  }
}

Collector::~Collector()
{
  chkError(cudaGraphExecDestroy(m_graphExec));

  if (m_metrics.rcvWtCtr.h) {
    chkError(cudaFreeHost(m_metrics.rcvWtCtr.h));
    m_metrics.rcvWtCtr.h = nullptr;
    m_metrics.rcvWtCtr.d = nullptr;
  }
  if (m_metrics.fwdWtCtr.h) {
    chkError(cudaFreeHost(m_metrics.fwdWtCtr.h));
    m_metrics.fwdWtCtr.h = nullptr;
    m_metrics.fwdWtCtr.d = nullptr;
  }

  if (m_metrics.state.h) {
    chkError(cudaFreeHost(m_metrics.state.h));
    m_metrics.state.h = nullptr;
    m_metrics.state.d = nullptr;
  }

  if (m_retCode_d)  chkError(cudaFree(m_retCode_d));
  m_retCode_d = nullptr;

  if (m_state_d)  chkError(cudaFree(m_state_d));
  m_state_d = nullptr;

  if (m_index_d)  chkError(cudaFree(m_index_d));
  m_index_d = nullptr;

  chkError(cudaStreamDestroy(m_stream));

  if (m_collectorQueue.d)  chkError(cudaFree(m_collectorQueue.d));
  delete m_collectorQueue.h;
  m_collectorQueue.d = nullptr;
  m_collectorQueue.h = nullptr;
}

int Collector::setupMetrics(const std::shared_ptr<MetricExporter> exporter,
                            std::map<std::string, std::string>&   labels)
{
  *m_metrics.state.h = 0;
  *m_metrics.rcvWtCtr.h = 0;
  *m_metrics.fwdWtCtr.h = 0;
  exporter->add("DRP_tigState", labels, MetricType::Gauge,   [&](){ return m_metrics.state.h    ? *m_metrics.state.h : 0; });
  exporter->add("DRP_tigRcv",   labels, MetricType::Counter, [&](){ return m_metrics.rcvWtCtr.h ? *m_metrics.rcvWtCtr.h : 0; });
  exporter->add("DRP_tigFwd",   labels, MetricType::Counter, [&](){ return m_metrics.fwdWtCtr.h ? *m_metrics.fwdWtCtr.h : 0; });

  m_metrics.pndWtCtr = 0;
  m_metrics.pidWtCtr = 0;
  exporter->add("DRP_colRcv",   labels, MetricType::Counter, [&](){ return m_metrics.pndWtCtr; });
  exporter->add("DRP_pidWtCtr", labels, MetricType::Counter, [&](){ return m_metrics.pidWtCtr; });

  exporter->add("DRP_colQueOcc", labels, MetricType::Gauge,   [&](){ return m_collectorQueue.h->occupancy(); });

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

int Collector::_setupGraph()
{
  // Build the graph
  logging::debug("Recording collector graph");
  cudaGraph_t graph = _recordGraph(m_stream);
  if (graph == 0) {
    return -1;
  }

  // Instantiate the graph
  if (chkError(cudaGraphInstantiate(&m_graphExec, graph, cudaGraphInstantiateFlagDeviceLaunch),
               "Collector graph create failed")) {
    return -1;
  }

  // No need to hang on to the stream info
  cudaGraphDestroy(graph);

  // Upload the graph so it can be launched by the scheduler kernel later
  logging::debug("Uploading Collector graph...");
  if (chkError(cudaGraphUpload(m_graphExec, m_stream), "Collector graph upload failed")) {
    return -1;
  }

  return 0;
}

// This kernel collects and event builds contributions from the DMA streams
static __global__
void _trgInpGenRcv(unsigned*      const __restrict__ state,
                   unsigned*      const __restrict__ index,
                   RingIndexDtoD* const __restrict__ inputQueue,
                   uint64_t*      const __restrict__ stateMon,
                   uint64_t*      const __restrict__ rcvWtCtr)
{
  //printf("### _trgInpGenRcv: tail %u, head %u\n", *tail, *head);

  if (*state == 0) {
    //*stateMon = 2;
    // Get the intermediate buffer index
    unsigned idx;
    unsigned ns{8};
    while (!inputQueue->pop(&idx)) {
      __nanosleep(ns);
      if (ns < 256)  ns *= 2;
      else {
        //*stateMon = 4;
        return;
      }
    }
    //printf("### _trgInpGenRcv: got idx %u\n", idx);
    *state = 1;
    //*stateMon = 5;
    //++(*rcvWtCtr);

    // Advance index
    *index = idx;
  }
}

// This will re-launch the current graph
static __global__
void _trgInpGenLoop(unsigned*      const  __restrict__ state,
                    unsigned*      const  __restrict__ index,
                    RingIndexDtoH* const  __restrict__ outputQueue,
                    uint64_t*      const  __restrict__ stateMon,
                    uint64_t*      const  __restrict__ fwdWtCtr,
                    cuda::std::atomic<unsigned> const& terminate)
{
  if (*state == 2) {
    //*stateMon = 10;
    // Push index to host and increment to the next one
    //printf("### _trgInpGenLoop: push index %u\n", *index);
    bool rc;
    unsigned ns{8};
    while ( (rc = !outputQueue->push(*index)) ) {
      __nanosleep(ns);
      if (ns < 256)  ns *= 2;
      else {
        //*stateMon = 11;
        break;
      }
    }
    if (!rc) {
      //printf("### Collector: pushed, new index %u\n", *index);
      *state = 0;
      //*stateMon = 12;
      //++(*fwdWtCtr);
    }
  }

  // This will re-launch the current graph
  //printf("### _trgInpGenLoop: relaunch\n");
  if (!terminate.load(cuda::std::memory_order_acquire))  {
    cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
  }
}

cudaGraph_t Collector::_recordGraph(cudaStream_t stream)
{
  col_scoped_range r{/*"Collector::_recordGraph"*/}; // Expose function name via NVTX

  auto hostWrtBufs_d  = m_pool.hostWrtBufs_d();
  auto hostWrtBufsCnt = m_pool.hostWrtBufsSize() / sizeof(*hostWrtBufs_d);
  auto calibBuffers   = m_pool.calibBuffers_d();
  auto calibBufsCnt   = m_pool.calibBufsSize() / sizeof(*calibBuffers);

  if (chkError(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "Collector stream begin capture failed")) {
    return 0;
  }

  // Find which pebble buffers are ready for processing
  _trgInpGenRcv<<<1, 1, 0, stream>>>(m_state_d,
                                     m_index_d,
                                     m_readerQueue.d,
                                     m_metrics.state.d,
                                     m_metrics.rcvWtCtr.d);

  // Process calibBuffers[tail] into TEB input data placed at the end of hostWrtBufs[tail]
  if (m_triggerPrimitive) { // else this DRP doesn't provide TEB input
    m_triggerPrimitive->event(stream,
                              m_state_d,
                              calibBuffers,
                              calibBufsCnt,
                              hostWrtBufs_d,
                              hostWrtBufsCnt,
                              m_index_d,
                              m_retCode_d);
  }

  // Post the buffer to the host and relaunch
  _trgInpGenLoop<<<1, 1, 0, stream>>>(m_state_d,
                                      m_index_d,
                                      m_collectorQueue.d,
                                      m_metrics.state.d,
                                      m_metrics.fwdWtCtr.d,
                                      m_terminate_d);

  cudaGraph_t graph;
  if (chkError(cudaStreamEndCapture(stream, &graph),
               "Collector stream end capture failed")) {
    return 0;
  }

  return graph;
}

void Collector::start()
{
  logging::info("Collector starting");

  resetEventCounter();

  // Launch the Collector graph
  printf("*** Collector: Launching graph\n");
  chkFatal(cudaGraphLaunch(m_graphExec, m_stream));
}

void Collector::_freeDma(unsigned index)
{
  //printf("*** Collector: Collector::freeDma: 1 idx %u\n", index);
  m_pebbleQueue.h->push(index);
  //printf("*** Collector: Collector::freeDma: 2 idx %u\n", index);

  m_pool.freeDma(1, nullptr);
  //printf("*** Collector: Collector::freeDma: 3 idx %u\n", index);
}

void Collector::freeDma(PGPEvent* event)
{
  //printf("*** Collector::freeDma: evt %p, idx %u, pgpIdx %zu\n", event, event->buffers[0].index, event - &m_pool.pgpEvents[0]);
  const uint32_t lane = 0;                   // The lane is always 0 for GPU-enabled PGP devices
  DmaBuffer* buffer = &event->buffers[lane];
  event->mask = 0;
  _freeDma(buffer->index);
}

bool Collector::receive(Detector* det)
{
  col_scoped_range r{/*"Collector::receive"*/}; // Expose function name via NVTX

#if defined(USE_TRACEBUFFER)
  struct trace_t
  {
    unsigned index;
    uint64_t pid;
    uint64_t lastPid;
  };
  static std::vector<struct trace_t> traceBuffer(2048);
  static unsigned itb = 0;
#endif

  const auto     hostWrtBufs    = m_pool.hostWrtBufs_h(); // When no error, hdrs in all are the same
  const auto     hostWrtBufsCnt = m_pool.hostWrtBufsSize() / sizeof(*hostWrtBufs);
  const uint32_t bufferMask     = m_pool.nbuffers() - 1;
  uint64_t       lastPid        = m_lastPid;

  unsigned index;
  while(!m_collectorQueue.h->pop(&index)) {
    if (m_terminate.load(std::memory_order_acquire)) [[unlikely]] {
      m_metrics.nDmaRet = 0;
      return false;
    }
  }
  //printf("*** Collector::receive: got index %u\n", index);
  ++m_metrics.pndWtCtr;

  col_scoped_range loop_range{/*"Collector::receive", */nvtx3::payload{index}};
  const auto dmaDsc       = (DmaDsc*)&hostWrtBufs[index * hostWrtBufsCnt];
  const auto timingHeader = (TimingHeader*)&dmaDsc[1];

  if (m_para.verbose > 2) {
    printf("*** Collector::receive: dmaDsc[%u] %p, th %p\n", index, dmaDsc, timingHeader);
    const auto& p = (const uint32_t*)dmaDsc;
    printf("*** Collector::receive: dmaBuf %08x %08x | %08x %08x %08x %08x %08x %08x\n",
           p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
  }

  // Wait for pulse ID to become non-zero
  uint64_t pid = timingHeader->pulseId();
  //printf("*** 1 pid %014lx\n", pid);
  while (pid <= lastPid) {
    col_scoped_range loop_range2{/*"Collector::receive wait"*/};
    if (m_terminate.load(std::memory_order_acquire)) [[unlikely]]  return false;
    pid = timingHeader->pulseId();
  }
  ++m_metrics.pidWtCtr;
  //printf("*** 2 pid %014lx\n", pid);

  nvtx3::mark("Collector received", nvtx3::payload{index});
#if defined(USE_TRACEBUFFER)
  traceBuffer[itb].tail    = tail;
  traceBuffer[itb].head    = head;
  traceBuffer[itb].pid     = pid;
  traceBuffer[itb].lastPid = lastPid;
  itb = (itb + 1) % traceBuffer.size();
#endif

  if (pid <= lastPid) [[unlikely]]
    logging::error("%s: PulseId did not advance: %014lx <= %014lx", __PRETTY_FUNCTION__, pid, lastPid);
  lastPid = pid;

  // Check for DMA buffer overflow
  if (dmaDsc->overflow()) [[unlikely]] {
    logging::critical("DMA buffer overflow: size %d B vs %d B", dmaDsc->size, m_pool.dmaSize());
    // @todo: Temporarily commented out: abort();
  }

  // Measure TimingHeader arrival latency as early as possible
  if (pid - m_latPid > 1300000/14) {    // 10 Hz
      m_metrics.latency = Eb::latency<us_t>(timingHeader->time);
      m_latPid = pid;
  }

#ifdef HOST_REARMS_DMA
  // Write to the DMA start register in the FPGA
  unsigned dmaIdx = index % m_pool.dmaCount();
  //printf("*** Collector::receive: Enable write to DMA buffer %u\n", dmaIdx);
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
  uint32_t lane = 0;      // The lane is always 0 for GPU-enabled PGP devices
  m_metrics.dmaSize   = size;
  m_metrics.dmaBytes += size;

  if (timingHeader->error()) [[unlikely]] {
      if (m_metrics.nTmgHdrError < 5) { // Limit prints at rate
          logging::error("Timing header error bit is set");
      }
      ++m_metrics.nTmgHdrError;
  }

  uint32_t evtCounter = timingHeader->evtCounter & EvtCtrMask;
  unsigned pgpIndex = evtCounter & bufferMask;
  PGPEvent* event = &m_pool.pgpEvents[pgpIndex];
  if (event->mask)  printf("*** Collector::receive: PGPEvent mask (%02x) != 0 for pgpIdx %u, ctr %u\n",
                           event->mask, pgpIndex, evtCounter);
  DmaBuffer* buffer = &event->buffers[lane]; // @todo: Do we care about this?
  buffer->size = size;                       //   "
  buffer->index = index;                     //   "
  event->mask |= (1 << lane);

  m_pool.allocateDma(); // DMA buffer was allocated when f/w incremented evtCounter

  // Check whether the DMA is reporting an error
  // @todo: C1100: if (dmaDsc->header ^ ~dmaDsc->errorMask()) [[unlikely]] {
  if (dmaDsc->header & dmaDsc->errorMask()) [[unlikely]] {    // Ignore SOF for KCU usage
    // Assume we can recover from non-overflow DMA errors
    if (m_metrics.nDmaErrors < 5) {     // Limit prints at rate
      logging::error("DMA error 0x%08x", dmaDsc->header);
    }
    // This assumes the DMA succeeded well enough that evtCounter is valid
    handleBrokenEvent(*event);
    freeDma(event);                     // Leaves event mask = 0
    ++m_metrics.nDmaErrors;
    return false;
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
    resetEventCounter();              // Compensate for the ClearReadout sent before BeginRun
  }
  if (evtCounter != ((m_lastComplete + 1) & EvtCtrMask)) [[unlikely]] {
    if (m_lastTid != TransitionId::Unconfigure) {
      if ((m_metrics.nPgpJumps < 5) || m_para.verbose) { // Limit prints at rate
        auto evtCntDiff = evtCounter - m_lastComplete;
        logging::error("%sPGPReader: Jump in TimingHeader evtCounter %u -> %u | difference %d, DMA size %u%s, index %u",
                       RED_ON, m_lastComplete, evtCounter, evtCntDiff, size, RED_OFF, index);
        logging::error("new data: %08x %08x %08x %08x %08x %08x  (%s)",
                       data[0], data[1], data[2], data[3], data[4], data[5], TransitionId::name(transitionId));
        logging::error("lastData: %08x %08x %08x %08x %08x %08x  (%s)",
                       m_lastData[0], m_lastData[1], m_lastData[2], m_lastData[3], m_lastData[4], m_lastData[5], TransitionId::name(m_lastTid));
      }
      handleBrokenEvent(*event);
      freeDma(event);                   // Leaves event mask = 0
      ++m_metrics.nPgpJumps;
#if defined(USE_TRACEBUFFER)
      for (unsigned i = 0; i < traceBuffer.size(); ++i) {
        unsigned j = (itb + i) % traceBuffer.size();
        auto& tb = traceBuffer[j];
        printf("%4u:%4u: t %4u h %4u p %014lx l %014lx\n", i, j, tb.tail, tb.head, tb.pid, tb.lastPid);
      }
      sleep(10);
      printf("cur: p %014lx, e %u\n", timingHeader->pulseId(), timingHeader->evtCounter & EvtCtrMask);
#endif
      abort();
      return false;                     // Throw away out-of-sequence events
    } else if (transitionId != TransitionId::Configure) {
      freeDma(event);                   // Leaves event mask = 0
      return false;                     // Drain
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
    freeDma(event);                     // Leaves event mask = 0
    ++m_metrics.nNoComRoG;
    return false;
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
      freeDma(event);                   // Leaves event mask = 0
      ++m_metrics.nMissingRoGs;
      return false;
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

  ++m_metrics.nEvents;
  //printf("*** Collector::receive: index %u, event->mask %02x\n", index, event->mask);

  m_metrics.nDmaRet = 1;
  return true;
}
