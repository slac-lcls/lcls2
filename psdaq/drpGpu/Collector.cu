#include "Collector.hh"

#include "Detector.hh"
#include "Reader.hh"

#include "psalg/utils/SysLog.hh"
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
                     const std::atomic<bool>&           terminate,
                     const cuda::std::atomic<unsigned>& terminate_d) :
  m_pool            (pool),
  m_triggerPrimitive(triggerPrimitive),
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

// This kernel collects and event builds contributions from the DMA streams
static __global__
void _collector(unsigned*             __restrict__ head,
                unsigned*             __restrict__ tail,
                RingIndexDtoD&                     readerQueue,
                const cuda::std::atomic<unsigned>& terminate)
{
  int panel = blockIdx.x * blockDim.x + threadIdx.x;
  //printf("### Collector: panel %u, tail %u, head %u\n", panel, *tail, *head);

  // Refresh the head if the tail has caught up to it
  // It might be desireable to refresh the head on every call, but that could
  // prevent progressing the tail toward the head since it blocks when there
  // is no change.  @todo: Revisit this
  if (*tail == *head) {
    __shared__ unsigned hd0;

    // Get one intermediate buffer index per FPGA
    unsigned hdN;
    while ((hdN = readerQueue.pend()) == *head) {
      if (terminate.load(cuda::std::memory_order_acquire))  return;
    }
    //printf("### Collector: panel %u, hdN %u\n", panel, hdN);
    if (panel == 0)  hd0 = hdN;

    // @todo: grp.sync();
    __syncthreads();

    if (hdN != hd0) {                   // Do this even for panel == 0?
      printf("Index mismatch for FPGA[%u]: %u != %u", panel, hdN, hd0);
      while (true);                     // abort(); ???
    }
    // Advance head
    if (panel == 0)  *head = hdN;
  }
}

// This will re-launch the current graph
static __global__
void _graphLoop(unsigned*                          idx,
                RingIndexDtoH&                     collectorQueue,
                const cuda::std::atomic<unsigned>& terminate)
{
  if (terminate.load(cuda::std::memory_order_acquire))  return;

  // Push index to host
  //printf("### Collector: post idx %u\n", *idx);
  *idx = collectorQueue.post(*idx);
  //printf("### Collector: posted, new idx %u\n", *idx);

  cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
}

cudaGraph_t Collector::_recordGraph(cudaStream_t stream)
{
  col_scoped_range r{/*"Collector::_recordGraph"*/}; // Expose function name via NVTX

  auto nPanels        = m_pool.panels().size();
  auto hostWrtBufs_d  = m_pool.hostWrtBufs_d();
  auto hostWrtBufsCnt = m_pool.hostWrtBufsSize() / sizeof(**hostWrtBufs_d);
  auto calibBuffers   = m_pool.calibBuffers_d();
  auto calibBufsCnt   = m_pool.calibBufsSize() / sizeof(*calibBuffers);

  if (chkError(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "Collector stream begin capture failed")) {
    return 0;
  }

  // Find which pebble buffers are ready for processing
  _collector<<<1, 1, 0, stream>>>(m_head,
                                  m_tail,
                                  *m_readerQueue.d,
                                  m_terminate_d);

  // Process calibBuffers[tail] into TEB input data placed at the end of hostWrtBufs[tail]
  // @todo: Deal with transitions
  if (m_triggerPrimitive) { // else this DRP doesn't provide TEB input
    m_triggerPrimitive->event(stream,
                              calibBuffers,
                              calibBufsCnt,
                              hostWrtBufs_d,
                              hostWrtBufsCnt,
                              *m_tail,
                              nPanels);
  }

  // Re-launch! Additional behavior can be put in graphLoop as needed. For now, it just re-launches the current graph.
  _graphLoop<<<1, 1, 0, stream>>>(m_tail, *m_collectorQueue.d, m_terminate_d);

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
  chkFatal(cudaGraphLaunch(m_graphExec, m_stream));
}

void Collector::freeDma(unsigned index)
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
  //printf("*** Collector::freeDma: evt %p\n", event);
  const uint32_t lane = 0;                   // The lane is always 0 for GPU-enabled PGP devices
  DmaBuffer* buffer = &event->buffers[lane];
  event->mask = 0;
  freeDma(buffer->index);
}

unsigned Collector::_checkDmaDsc(unsigned index) const
{
  unsigned   rc   = 0;
  const auto cnt  = m_pool.hostWrtBufsSize() / sizeof(uint32_t);
  const auto dsc0 = (DmaDsc*)(&m_pool.hostWrtBufsVec_h()[0][index * cnt]);

  logging::debug("panel %d: dma %d hdr: err %08x,  sz %08x, rsvd %08x %08x %08x %08x %08x %08x",
                 0, index, dsc0->error, dsc0->size, dsc0->_rsvd[0], dsc0->_rsvd[1], dsc0->_rsvd[2],
                 dsc0->_rsvd[3], dsc0->_rsvd[4], dsc0->_rsvd[5]);

  for (unsigned i = 1; i < m_pool.panels().size(); ++i) {
    bool ne = false;
    const auto dscN = (DmaDsc*)(&m_pool.hostWrtBufsVec_h()[i][index * cnt]);
    ne |= dscN->error != dsc0->error;
    ne |= dscN->size  != dsc0->size;

    if (ne) {
      if (rc ^ 1) {
        logging::debug("panel %d: dma %d hdr: err %08x,  sz %08x, rsvd %08x %08x %08x %08x %08x %08x",
                       0, index, dsc0->error, dsc0->size, dsc0->_rsvd[0], dsc0->_rsvd[1], dsc0->_rsvd[2],
                       dsc0->_rsvd[3], dsc0->_rsvd[4], dsc0->_rsvd[5]);
      }
      logging::debug("panel %d: idx %d dma: err %08x,  sz %08x, rsvd %08x %08x %08x %08x %08x %08x",
                     i, index, dscN->error, dscN->size, dscN->_rsvd[0], dscN->_rsvd[1], dscN->_rsvd[2],
                     dscN->_rsvd[3], dscN->_rsvd[4], dscN->_rsvd[5]);
      rc |= 1;                          // If different, include panel 0 in the list
    }
    rc |= 1 << i;
  }

  return rc;
}

unsigned Collector::_checkTimingHeader(unsigned index) const
{
  unsigned   rc   = 0;
  const auto cnt  = m_pool.hostWrtBufsSize() / sizeof(uint32_t);
  const auto dsc0 = (DmaDsc*)(&m_pool.hostWrtBufsVec_h()[0][index * cnt]);
  const auto th0  = (TimingHeader*)&dsc0[1];

  logging::debug("panel %d: idx %d  th: ctl %02x, pid %014lx, ts %016lx, env %08x, ctr %08x, opq %08x %08x",
                 0, index, th0->control(), th0->pulseId(), th0->time.value(), th0->env, th0->evtCounter,
                 th0->_opaque[0], th0->_opaque[1]);

  for (unsigned i = 1; i < m_pool.panels().size(); ++i) {
    bool ne = false;
    const auto dscN = (DmaDsc*)(&m_pool.hostWrtBufsVec_h()[i][index * cnt]);
    const auto thN  = (TimingHeader*)&dscN[1];
    ne |= thN->control()    != th0->control();
    ne |= thN->pulseId()    != th0->pulseId();
    ne |= thN->time.value() != th0->time.value();
    ne |= thN->env          != th0->env;
    ne |= thN->evtCounter   != th0->evtCounter;

    if (ne) {
      if (rc ^ 1) {
        logging::debug("panel %d: idx %d  th: ctl %02x, pid %014lx, ts %016lx, env %08x, ctr %08x, opq %08x %08x",
                       0, index, th0->control(), th0->pulseId(), th0->time.value(), th0->env, th0->evtCounter,
                       th0->_opaque[0], th0->_opaque[1]);
      }
      logging::debug("panel %d: dma %d  th: ctl %02x, pid %014lx, ts %u.%09u, env %08x, ctr %08x, opq %08x %08x",
                     i, index, thN->control(), thN->pulseId(), thN->time.seconds(), thN->time.nanoseconds(),
                     thN->env, thN->evtCounter, thN->_opaque[0], thN->_opaque[1]);
      rc |= 1;                          // If different, include panel 0 in the list
    }
    rc |= 1 << i;
  }

  return rc;
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

  const auto     hostWrtBufs    = m_pool.hostWrtBufsVec_h()[0]; // When no error, hdrs in all are the same
  const auto     hostWrtBufsCnt = m_pool.hostWrtBufsSize() / sizeof(*hostWrtBufs);
  const uint32_t bufferMask     = m_pool.nbuffers() - 1;
  uint64_t       lastPid        = m_lastPid;

  unsigned head = m_collectorQueue.h->pend();
  unsigned tail = m_last;
  //if (tail != head)  printf("*** Collector::receive: tail %u, head %u\n", tail, head);
  while (tail != head) {
    //printf("*** Collector::receive: tail %u\n", tail);
    col_scoped_range loop_range{/*"Collector::receive", */nvtx3::payload{tail}};
    const auto dmaDsc       = (DmaDsc*)(&hostWrtBufs[tail * hostWrtBufsCnt]);
    const auto timingHeader = (TimingHeader*)&dmaDsc[1];

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
    if (dmaDsc->error & 0x4) [[unlikely]] {
      logging::critical("DMA buffer overflow: size %d B", m_pool.dmaSize());
      exit(EXIT_FAILURE);
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
    for (auto& panel : m_pool.panels()) {
      auto rc = gpuSetWriteEn(panel.gpu.fd(), dmaIdx);
      if (rc < 0) [[unlikely]] {
        logging::critical("Failed to reenable buffer %u for write: %zd: %m", dmaIdx, rc);
        abort();
      }
    }
#endif

    // Handle the case when the headers don't match across panels
    // @todo: Too expensive?  This fetches the headers of all panels from the GPU
    //        Maybe do the test on the GPU and set a flag if they differ and
    //        print here when it is set
    unsigned dmas, ths;
    if ( (dmas = _checkDmaDsc(tail)) || (ths = _checkTimingHeader(tail)) ) [[unlikely]] {
      // Assume we can recover from non-matching panel headers
      logging::error("Headers differ between panels: DmaDsc: %02x, TimingHeader: %02x", dmas, ths);
      freeDma(tail);                    // Leaves event mask = 0
      metrics.m_nHdrMismatch += 1;
      continue;
    }

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
    PGPEvent*  event  = &m_pool.pgpEvents[pgpIndex];
    //if (event->mask)  printf("*** Collector::receive: PGPEvent mask (%02x) != 0 for ctr %u\n",
    //                         event->mask, pgpIndex);
    DmaBuffer* buffer = &event->buffers[lane]; // @todo: Do we care about this?
    buffer->size = size;                       //   "
    buffer->index = index;                     //   "
    event->mask |= (1 << lane);

    m_pool.allocateDma(); // DMA buffer was allocated when f/w incremented evtCounter

    if (dmaDsc->error) [[unlikely]] {
      // Assume we can recover from non-overflow DMA errors
      if (metrics.m_nDmaErrors < 5) {   // Limit prints at rate
        logging::error("DMA error 0x%x", dmaDsc->error);
      }
      // This assumes the DMA succeeded well enough that evtCounter is valid
      handleBrokenEvent(*event);
      freeDma(event);                   // Leaves event mask = 0
      metrics.m_nDmaErrors += 1;
      continue;
    }

    XtcData::TransitionId::Value transitionId = timingHeader->service();
    const uint32_t* data = reinterpret_cast<const uint32_t*>(timingHeader);
    logging::debug("PGPReader  size %u  hdr %016lx.%016lx.%08x  err 0x%x",
                   size,
                   reinterpret_cast<const uint64_t*>(data)[0], // PulseId
                   reinterpret_cast<const uint64_t*>(data)[1], // Timestamp
                   reinterpret_cast<const uint32_t*>(data)[4], // env
                   dmaDsc->error);

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
        logging::info("PGPReader  saw %s @ %u.%09u (%014lx)",
                      XtcData::TransitionId::name(transitionId),
                      timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                      timingHeader->pulseId());
      }
      else {
        logging::debug("PGPReader  saw %s @ %u.%09u (%014lx)",
                       XtcData::TransitionId::name(transitionId),
                       timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                       timingHeader->pulseId());
      }
    }

    metrics.m_nevents += 1;
    tail = (tail + 1) & bufferMask;
    //printf("*** Collector::receive: tail %u\n", tail);
  }
  unsigned nEvents = (head - m_last) & bufferMask;
  m_last = tail;
  m_lastPid = lastPid;

  //if (nEvents)  printf("*** Collector::receive: head %u, nEvents %u\n", head, nEvents);
  return nEvents;
}
