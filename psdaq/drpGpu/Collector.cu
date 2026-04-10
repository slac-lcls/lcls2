#include "Collector.hh"

#include "Detector.hh"
#include "Reader.hh"

#include "psalg/utils/SysLog.hh"
#include "psdaq/service/EbDgram.hh"     // For TimingHeader
#include "psdaq/trigger/TriggerPrimitive.hh"
#include "psdaq/trigger/tmoTebPrimitive_gpu_dev.hh"
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
  m_gpuDispatchType (triggerPrimitive ? triggerPrimitive->gpuDispatchType() : Trg::GpuDispatchType::None),
  m_terminate       (terminate),
  m_terminate_d     (terminate_d),
  m_readerQueue     (reader->queue()), // Buffer index queue for Reader to Collector comms
  m_last            (0),
  m_lastPid         (0),
  m_latPid          (0),
  m_lastComplete    (0),
  m_lastTid         (TransitionId::Unconfigure),
  m_para            (para)
{
  // Set up buffer index queue for Collector to Host comms
  m_collectorQueue.h = new RingIndexDtoH(pool.nbuffers(), m_terminate, m_terminate_d);
  chkError(cudaMalloc(&m_collectorQueue.d,                     sizeof(*m_collectorQueue.d)));
  chkError(cudaMemcpy( m_collectorQueue.d, m_collectorQueue.h, sizeof(*m_collectorQueue.d), cudaMemcpyHostToDevice));

  // Get the range of priorities available [ greatest_priority, lowest_priority ]
  int prioLo;
  int prioHi;
  chkError(cudaDeviceGetStreamPriorityRange(&prioLo, &prioHi));
  int prio{prioLo-1};
  logging::debug("Collector stream priority (range: LOW: %d to HIGH: %d): %d", prioLo, prioHi, prio);

  // Create the Collector EB stream with higher priority than the Reader
  chkFatal(cudaStreamCreateWithPriority(&m_stream, cudaStreamNonBlocking, prio));
  //chkFatal(cudaExecutionCtxStreamCreate(&m_stream, green_ctx, cudaStreamDefault, prio));
  logging::debug("Done with creating collector stream");

  // Keep track of the head and tail indices of the Collector stream
  chkError(cudaMalloc(&m_head,    sizeof(*m_head)));
  chkError(cudaMemset( m_head, 0, sizeof(*m_head)));
  chkError(cudaMalloc(&m_tail,    sizeof(*m_tail)));
  chkError(cudaMemset( m_tail, 0, sizeof(*m_tail)));

  // Prepare the Collector graph
  if (_setupGraph()) {
    logging::critical("Failed to set up Collector graph");
    abort();
  }
}

Collector::~Collector()
{
  chkError(cudaGraphExecDestroy(m_graphExec));

  chkError(cudaFree(m_tail));
  chkError(cudaFree(m_head));

  chkError(cudaStreamDestroy(m_stream));

  chkError(cudaFree(m_collectorQueue.d));
  delete m_collectorQueue.h;
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

// This device function collects and event builds contributions from the DMA streams
static __device__
void _collectorStep(unsigned*      const  __restrict__ head,
                    unsigned*      const  __restrict__ tail,
                    RingIndexDtoD* const  __restrict__ readerQueue,
                    cuda::std::atomic<unsigned> const& terminate)
{
  //printf("### Collector: tail %u, head %u\n", tail, head);

  // Refresh the head if the tail has caught up to it
  // It might be desireable to refresh the head on every call, but that could
  // prevent progressing the tail toward the head since it blocks when there
  // is no change.  @todo: Revisit this
  if (*tail == *head) {
    // Get the intermediate buffer index
    unsigned hd;
    unsigned ns{8};
    while ((hd = readerQueue->pend()) == *head) {
      if (terminate.load(cuda::std::memory_order_acquire))
        return;
      __nanosleep(ns);
      if (ns < 256)  ns *= 2;
    }
    //printf("### Collector: hd %u\n", hd);

    // Advance head
    *head = hd;
  }
}

// This will re-launch the current graph
static __device__
void _graphLoopStep(unsigned*      const  __restrict__ idx,
                    RingIndexDtoH* const  __restrict__ collectorQueue,
                    cuda::std::atomic<unsigned> const& terminate)
{
  if (terminate.load(cuda::std::memory_order_acquire))  return;

  // Push index to host and increment to the next one
  //printf("### Collector: post idx %u\n", *idx);
  *idx = collectorQueue->post(*idx);
  //printf("### Collector: posted, new idx %u\n", *idx);

  // This will re-launch the current graph
  //printf("### Collector: relaunch\n");
  cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
}

struct NoEventFn
{
  __device__
  void operator()(float     const* const,
                  const size_t,
                  uint32_t* const,
                  const size_t,
                  const unsigned) const
  {
  }
};

template<typename EventFn>
static __global__
void _fusedCollector(unsigned*        const __restrict__ head,
                     unsigned*        const __restrict__ tail,
                     RingIndexDtoD*   const __restrict__ readerQueue,
                     RingIndexDtoH*   const __restrict__ collectorQueue,
                     cuda::std::atomic<unsigned> const&  terminate,
                     EventFn                             eventFn,
                     float     const* const __restrict__ calibBuffers,
                     size_t           const              calibBufsCnt,
                     uint32_t*        const __restrict__ out,
                     size_t           const              outBufsCnt)
{
  // Find which pebble buffers are ready for processing
  _collectorStep(head, tail, readerQueue, terminate);

  if (terminate.load(cuda::std::memory_order_acquire))  return;

  // Process calibBuffers[tail] into TEB input data placed at the end of hostWrtBufs[tail]
  eventFn(calibBuffers, calibBufsCnt, out, outBufsCnt, *tail);

  // Re-launch! Additional behavior can be put in graphLoop as needed. For now, it just re-launches the current graph.
  _graphLoopStep(tail, collectorQueue, terminate);
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

  unsigned threads = 1;
  unsigned blocks  = 1;
  printf("Collector blocks %u * threads %u = %u threads\n", blocks, threads, blocks * threads);
  switch (m_gpuDispatchType) {
    case Trg::GpuDispatchType::TmoTeb:
      _fusedCollector<<<blocks, threads, 0, stream>>>(m_head,
                                                      m_tail,
                                                      m_readerQueue.d,
                                                      m_collectorQueue.d,
                                                      m_terminate_d,
                                                      Trg::TmoTebEventFn{},
                                                      calibBuffers,
                                                      calibBufsCnt,
                                                      hostWrtBufs_d,
                                                      hostWrtBufsCnt);
      break;
    case Trg::GpuDispatchType::None:
    default:
      _fusedCollector<<<blocks, threads, 0, stream>>>(m_head,
                                                      m_tail,
                                                      m_readerQueue.d,
                                                      m_collectorQueue.d,
                                                      m_terminate_d,
                                                      NoEventFn{},
                                                      calibBuffers,
                                                      calibBufsCnt,
                                                      hostWrtBufs_d,
                                                      hostWrtBufsCnt);
      break;
  }

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
  m_collectorQueue.h->release(index);
  //printf("*** Collector: Collector::freeDma: 2 idx %u\n", index);

  //printf("*** Collector: Collector::freeDma: 2 idx %u\n", index);
  m_readerQueue.h->release(index);
  //printf("*** Collector: Collector::freeDma: 3 idx %u\n", index);

  m_pool.freeDma(1, nullptr);
  //printf("*** Collector: Collector::freeDma: 4 idx %u\n", index);
}

void Collector::freeDma(PGPEvent* event)
{
  //printf("*** Collector::freeDma: evt %p, idx %u, pgpIdx %zu\n", event, event->buffers[0].index, event - &m_pool.pgpEvents[0]);
  const uint32_t lane = 0;                   // The lane is always 0 for GPU-enabled PGP devices
  DmaBuffer* buffer = &event->buffers[lane];
  event->mask = 0;
  _freeDma(buffer->index);
}

unsigned Collector::receive(Detector* det, CollectorMetrics& metrics)
{
  col_scoped_range r{/*"Collector::receive"*/}; // Expose function name via NVTX

#if defined(USE_TRACEBUFFER)
  struct trace_t
  {
    unsigned tail;
    unsigned head;
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

  unsigned head = m_collectorQueue.h->pend();
  unsigned tail = m_last;
  //if (tail != head)  printf("*** Collector::receive: tail %u, head %u\n", tail, head);
  while (tail != head) {
    //printf("*** Collector::receive: tail %u\n", tail);
    col_scoped_range loop_range{/*"Collector::receive", */nvtx3::payload{tail}};
    const auto dmaDsc       = (DmaDsc*)&hostWrtBufs[tail * hostWrtBufsCnt];
    const auto timingHeader = (TimingHeader*)&dmaDsc[1];

    if (m_para.verbose) {
      printf("*** Collector::receive: dmaDsc[%u] %p, th %p\n", tail, dmaDsc, timingHeader);
      const auto& p = (const uint32_t*)dmaDsc;
      printf("*** Collector::receive: dmaBuf %08x %08x | %08x %08x %08x %08x %08x %08x\n",
             p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
    }

    // Wait for pulse ID to become non-zero
    uint64_t pid = timingHeader->pulseId();
    //printf("*** 1 pid %014lx\n", pid);
    while (pid <= lastPid) {
      col_scoped_range loop_range2{/*"Collector::receive wait"*/};
      if (m_terminate.load(std::memory_order_acquire)) [[unlikely]]  break;
      pid = timingHeader->pulseId();
    }
    if (m_terminate.load(std::memory_order_acquire)) [[unlikely]]  break;
    //printf("*** 2 pid %014lx\n", pid);

    nvtx3::mark("Collector received", nvtx3::payload{tail});
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
    if (pid - m_latPid > 1300000/14) { // 10 Hz
        metrics.m_latency = Eb::latency<us_t>(timingHeader->time);
        m_latPid = pid;
    }

#ifdef HOST_REARMS_DMA
    // Write to the DMA start register in the FPGA
    unsigned dmaIdx = tail % m_pool.dmaCount();
    //printf("*** Collector::receive: Enable write to DMA buffer %u\n", dmaIdx);
    auto rc = gpuSetWriteEn(m_pool.panel->gpu.fd(), dmaIdx);
    if (rc < 0) [[unlikely]] {
      logging::critical("Failed to reenable buffer %u for write: %zd: %m", dmaIdx, rc);
      abort();
    }
#endif

    logging::debug("dma %u, hdr %08x, sz %08x = %u", tail, dmaDsc->header, dmaDsc->size);
    logging::debug("idx %u  th: ctl %02x, pid %014lx, ts %016lx, env %08x, ctr %08x, opq %08x %08x",
                   tail, timingHeader->control(), timingHeader->pulseId(), timingHeader->time.value(),
                   timingHeader->env, timingHeader->evtCounter,
                   timingHeader->_opaque[0], timingHeader->_opaque[1]);

    uint32_t size = dmaDsc->size;       // Size of the DMA
    uint32_t index = tail;
    uint32_t lane = 0;      // The lane is always 0 for GPU-enabled PGP devices
    metrics.m_dmaSize   = size;
    metrics.m_dmaBytes += size;

    if (timingHeader->error()) [[unlikely]] {
        if (metrics.m_nTmgHdrError < 5) { // Limit prints at rate
            logging::error("Timing header error bit is set");
        }
        metrics.m_nTmgHdrError += 1;
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

    // @todo: Need to work out which bits mean error
    if (dmaDsc->header) [[unlikely]] {  // @todo: Revisit bits to be ignored
      // Assume we can recover from non-overflow DMA errors
      if (metrics.m_nDmaErrors < 5) {   // Limit prints at rate
        logging::error("DMA error 0x%08x", dmaDsc->header);
      }
      // This assumes the DMA succeeded well enough that evtCounter is valid
// @todo: Temporarily commented out:
//      handleBrokenEvent(*event);
//      freeDma(event);                   // Leaves event mask = 0
//      metrics.m_nDmaErrors += 1;
//      continue;
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
        if ((metrics.m_nPgpJumps < 5) || m_para.verbose) { // Limit prints at rate
          auto evtCntDiff = evtCounter - m_lastComplete;
          logging::error("%sPGPReader: Jump in TimingHeader evtCounter %u -> %u | difference %d, DMA size %u%s, tail %u, head %u",
                         RED_ON, m_lastComplete, evtCounter, evtCntDiff, size, RED_OFF, tail, head);
          logging::error("new data: %08x %08x %08x %08x %08x %08x  (%s)",
                         data[0], data[1], data[2], data[3], data[4], data[5], TransitionId::name(transitionId));
          logging::error("lastData: %08x %08x %08x %08x %08x %08x  (%s)",
                         m_lastData[0], m_lastData[1], m_lastData[2], m_lastData[3], m_lastData[4], m_lastData[5], TransitionId::name(m_lastTid));
        }
        handleBrokenEvent(*event);
        freeDma(event);                 // Leaves event mask = 0
        metrics.m_nPgpJumps += 1;
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
        continue;                       // Throw away out-of-sequence events
      } else if (transitionId != TransitionId::Configure) {
        freeDma(event);                 // Leaves event mask = 0
        //abort();
        continue;                       // Drain
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
      freeDma(event);                   // Leaves event mask = 0
      metrics.m_nNoComRoG += 1;
      continue;
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
        freeDma(event);                 // Leaves event mask = 0
        metrics.m_nMissingRoGs += 1;
        continue;
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

    metrics.m_nevents += 1;
    tail = (tail + 1) & bufferMask;
    //printf("*** Collector::receive: tail %u, event->mask %02x\n", tail, event->mask);
  }
  unsigned nEvents = (head - m_last) & bufferMask;
  m_last = tail;
  m_lastPid = lastPid;

  //if (nEvents)  printf("*** Collector::receive: head %u, nEvents %u\n", head, nEvents);
  return nEvents;
}
