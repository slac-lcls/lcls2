#include "Collector.hh"

#include "Detector.hh"
#include "Reader.hh"

#include "psalg/utils/SysLog.hh"
#include "psdaq/service/EbDgram.hh"     // For TimingHeader
#include "psdaq/trigger/TriggerPrimitive.hh"
#include "psdaq/eb/eb.hh"

using namespace XtcData;
using namespace Pds;
using namespace Drp;
using namespace Drp::Gpu;
using logging = psalg::SysLog;
using us_t = std::chrono::microseconds;

static const char* const RED_ON  = "\033[0;31m";
static const char* const RED_OFF = "\033[0m";
static const unsigned EvtCtrMask = 0xffffff;


Collector::Collector(const Parameters&         para,
                     MemPoolGpu&               pool,
                     std::vector<Reader>&      readers,
                     Trg::TriggerPrimitive*    triggerPrimitive,
                     const std::atomic<bool>&  terminate_h,
                     const cuda::atomic<int>&  terminate_d) :
  m_pool            (pool),
  m_triggerPrimitive(triggerPrimitive),
  m_terminate_h     (terminate_h),
  m_terminate_d     (terminate_d),
  m_graph           (0),
  m_last            (0),
  m_lastPid         (0),
  m_latPid          (0),
  m_lastComplete    (0),
  m_lastTid         (TransitionId::Unconfigure),
  m_para            (para)
{
  // Gather buffer index queues for Reader to Collector comms
  chkError(cudaMalloc(&m_readerQueues_d, readers.size() * sizeof(*m_readerQueues_d)));
  for (unsigned i = 0; i < readers.size(); ++i) {
    chkError(cudaMemcpy(&m_readerQueues_d[i], readers[i].queue(), sizeof(*m_readerQueues_d), cudaMemcpyHostToDevice));
  }

  // Set up buffer index queue for Collector to Host comms
  m_collectorQueue.h = new Gpu::RingIndexDtoH(pool.nbuffers(), m_terminate_h, m_terminate_d);
  chkError(cudaMalloc(&m_collectorQueue.d,                     sizeof(*m_collectorQueue.d)));
  chkError(cudaMemcpy( m_collectorQueue.d, m_collectorQueue.h, sizeof(*m_collectorQueue.d), cudaMemcpyHostToDevice));

  /** Create the Collector EB stream **/
  chkFatal(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));
  logging::debug("Done with creating collector stream");

  // Set up a done flag to cache m_terminate's value and avoid some PCIe transactions
  chkError(cudaMalloc(&m_done,    sizeof(*m_done)));
  chkError(cudaMemset( m_done, 0, sizeof(*m_done)));

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
  printf("*** Collector dtor 1\n");
  chkError(cudaGraphExecDestroy(m_graphExec));
  chkError(cudaGraphDestroy(m_graph)); // @todo: Goes away?
  printf("*** Collector dtor 2\n");

  chkError(cudaFree(m_tail));
  printf("*** Collector dtor 2a\n");
  chkError(cudaFree(m_head));
  printf("*** Collector dtor 2b\n");
  chkError(cudaFree(m_done));
  printf("*** Collector dtor 3\n");

  chkError(cudaStreamDestroy(m_stream));
  printf("*** Collector dtor 4\n");

  chkError(cudaFree(m_collectorQueue.d));
  delete m_collectorQueue.h;
  printf("*** Collector dtor 5\n");

  chkError(cudaFree(m_readerQueues_d));
  printf("*** Collector dtor 6\n");
}

int Collector::_setupGraph()
{
  printf("*** Collector setupGraph 1\n");
  // Build the graph
  if (m_graph == 0) {        // @todo: Graphs can be created on the stack
    logging::debug("Recording collector graph");
    auto& hostWriteBufs = m_pool.hostWrtBufsVec_h();
    for (unsigned panel = 0; panel < m_pool.panels().size(); ++panel) {
      printf("*** Collector setupGraph attach 1: panel %u, sz %zu\n", panel, m_collectorQueue.h->size());
      // hostWriteBufs[panel][0] is the base pointer for the entire allocation, i.e., all nBuffers
      chkError(cudaStreamAttachMemAsync(m_stream, hostWriteBufs[panel][0], 0, cudaMemAttachHost));
      printf("*** Collector setupGraph attach 2\n");
    }
    printf("*** Collector setupGraph 2\n");
    m_graph = _recordGraph(m_stream);
    if (m_graph == 0)
      return -1;
  }
  printf("*** Collector setupGraph 3\n");

  // Instantiate the graph
  if (chkError(cudaGraphInstantiate(&m_graphExec, m_graph, cudaGraphInstantiateFlagDeviceLaunch),
               "Collector graph create failed")) {
    return -1;
  }
  printf("*** Collector setupGraph 4\n");

  // @todo: No need to hang on to the stream info
  //cudaGraphDestroy(m_graph);

  // Upload the graph so it can be launched by the scheduler kernel later
  logging::debug("Uploading Collector graph...");
  if (chkError(cudaGraphUpload(m_graphExec, m_stream), "Collector graph upload failed")) {
    return -1;
  }
  printf("*** Collector setupGraph 5\n");

  return 0;
}

// This kernel collects and event builds contributions from the DMA streams
static __global__ void _collector(unsigned*         __restrict__ head,
                                  unsigned*         __restrict__ tail,
                                  RingIndexDtoD*    __restrict__ readerQueues,
                                  RingIndexDtoH&                 collectorQueue,
                                  uint32_t** const* __restrict__ in,
                                  const cuda::atomic<int>&       terminate,
                                  bool*             __restrict__ done)
{
  printf("*** _collector 1 tail %u, head %u\n", *tail, *head);
  int panel = blockIdx.x * blockDim.x + threadIdx.x;

  // Refresh the head if the tail has caught up to it
  // It might be desireable to refresh the head on every call, but that could
  // prevent progressing the tail toward the head since it blocks when there
  // is no change.  @todo: Revisit this
  if (*tail == *head) {
    printf("*** _collector 2\n");
    __shared__ unsigned hd0;

    // Get one intermediate buffer index per FPGA
    unsigned hdN;
    while ((hdN = readerQueues[panel].consume()) == *head) { // This can block
      if ( (*done = terminate.load(cuda::memory_order_acquire)) )  return;
    }
    printf("*** _collector 3, hdN %u\n", hdN);
    if (panel == 0)  hd0 = hdN;

    printf("*** _collector 4, hd0 %u\n", hd0);
    // @todo: grp.sync();
    __syncthreads();
    printf("*** _collector 5\n");

    if (hdN != hd0) {                   // Do this even for panel == 0?
      printf("Index mismatch for FPGA[%u]: %u != %u", panel, hdN, hd0);
      return;                           // abort(); ???
    }
    // Advance head
    if (panel == 0) *head = hdN;
  }

  printf("*** _collector 6, tail %u, head %u\n", *tail, *head);
  printf("*** _collector 6a, in %p\n", in);
  printf("*** _collector 6b, in[%u] %p\n", panel, in[panel]);
  printf("*** _collector 6c, in[%u][%u] %p\n", panel, *tail, in[panel][*tail]);
  printf("*** _collector 6d, in[%u][%u][0] %08x\n", panel, *tail, in[panel][*tail][0]);
  printf("*** _collector 6e, in[%u][%u][1] %08x\n", panel, *tail, in[panel][*tail][1]);
  printf("*** _collector 6f, in[%u][%u][8] %08x\n", panel, *tail, in[panel][*tail][8]);
  printf("*** _collector 6g, in[%u][%u][9] %08x\n", panel, *tail, in[panel][*tail][9]);

  // Check that the Pulse ID is the same for all FPGAs
  const unsigned  thOs = sizeof(DmaDsc) / sizeof(***in);
  const uint64_t& pid0 = *(uint64_t*)(&in[    0][*tail][thOs]);
  const uint64_t& pidN = *(uint64_t*)(&in[panel][*tail][thOs]);
  if (pidN != pid0) {
    // @todo: These should be counted these instead of printed...
    printf("Pulse ID mismatch for FPGA[%u] @ index %u: %014lx != %014lx", panel, *tail, pidN, pid0);
    return;                             // abort(); ???
  }
  printf("*** _collector 7, pid %014lx, env %08x\n", pid0, in[0][*tail][thOs+4]);

  // @todo: Copy only one device's DmaDsc and TimingHeader to the host?
  //        Currently, these are in managed memory, which resides on the device
  //        but with addresses the host can access.  Perhaps the transfer over
  //        PCIe is done only when the host does such an access, in which case
  //        there would seem to be no benefit to keeping these structures in
  //        non-managed device memory and then memcpying one of them to the host.
}

// This will re-launch the current graph
static __global__ void _graphLoop(unsigned*      idx,
                                  RingIndexDtoH& collectorQueue,
                                  const bool&    done)
{
  printf("*** Collector graphLoop 1\n");
  if (done)  return;
  printf("*** Collector graphLoop 1a, idx %u\n", *idx);

  // Push index to host
  *idx = collectorQueue.produce(*idx);
  printf("*** Collector graphLoop 2, idx %u\n", *idx);

  cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
  printf("*** Collector graphLoop 3\n");
}

cudaGraph_t Collector::_recordGraph(cudaStream_t& stream)
{
  printf("*** Collector::record 1\n");
  auto hostWrtBufs_d = m_pool.hostWrtBufs_d();
  auto calibBuffers  = m_pool.calibBuffers_d();

  printf("*** Collector::record 2\n");
  if (chkError(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "Collector stream begin capture failed")) {
    return 0;
  }
  printf("*** Collector::record 3, head %p, tail %p\n", m_head, m_tail);

  // Collect and event build data from the PGP FPGAs
  _collector<<<1, m_pool.panels().size(), 1, stream>>>(m_head,
                                                       m_tail,
                                                       m_readerQueues_d,
                                                       *m_collectorQueue.d,
                                                       hostWrtBufs_d,
                                                       m_terminate_d,
                                                       m_done);
  printf("*** Collector::record 4, trgPrmtv %p\n", m_triggerPrimitive);

  // Process calibBuffers[tail] into TEB input data placed at the end of hostWriteBufs[tail]
  // @todo: Deal with transitions
  // @todo: Provide a GPU-enabled base class for TriggerPrimitive
  if (m_triggerPrimitive) { // else this DRP doesn't provide TEB input
    m_triggerPrimitive->event(stream,
                              calibBuffers,
                              hostWrtBufs_d,
                              *m_tail,
                              *m_done);
  }
  printf("*** Collector::record 5\n");

  // Re-launch! Additional behavior can be put in graphLoop as needed. For now, it just re-launches the current graph.
  _graphLoop<<<1, 1, 0, stream>>>(m_tail, *m_collectorQueue.d, *m_done);
  printf("*** Collector::record 6\n");

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
  chkError(cuCtxSetCurrent(m_pool.context().context()));  // Needed, else kernels misbehave

  resetEventCounter();

  // Launch the Collector graph
  chkFatal(cudaGraphLaunch(m_graphExec, m_stream));
}

void Collector::freeDma(unsigned index)
{
  m_collectorQueue.h->release(index);
  // @todo: Make the host version accessible
  //for (unsigned i = 0; i < m_pool.panels().size(); ++i) {
  //  m_readerQueues_h[i]->release(index);
  //}

  m_pool.freeDma(1, nullptr);
}

void Collector::freeDma(PGPEvent* event)
{
  const uint32_t lane = 0;                   // The lane is always 0 for GPU-enabled PGP devices
  DmaBuffer* buffer = &event->buffers[lane];
  event->mask = 0;
  freeDma(buffer->index);
}

unsigned Collector::_checkDmaDsc(unsigned index) const
{
  unsigned rc = 0;
  const auto dsc0 = (DmaDsc*)(m_pool.hostWrtBufsVec_h()[0][index]);

  logging::debug("panel %d: dma %d hdr: err %08x,  sz %08x, rsvd %08x %08x %08x %08x %08x %08x",
                 0, index, dsc0->error, dsc0->size, dsc0->_rsvd[0], dsc0->_rsvd[1], dsc0->_rsvd[2],
                 dsc0->_rsvd[3], dsc0->_rsvd[4], dsc0->_rsvd[5]);

  for (unsigned i = 1; i < m_pool.panels().size(); ++i) {
    bool ne = false;
    const auto dscN = (DmaDsc*)(m_pool.hostWrtBufsVec_h()[i][index]);
    ne |= dscN->error != dsc0->error;
    ne |= dscN->size  != dsc0->size;

    if (ne) {
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
  unsigned rc = 0;
  const auto dsc0 = (DmaDsc*)(m_pool.hostWrtBufsVec_h()[0][index]);
  const auto th0  = (TimingHeader*)&dsc0[1];

  logging::debug("panel %d: idx %d  th: ctl %02x, pid %014lx, ts %016lx, env %08x, ctr %08x, opq %08x %08x",
                 0, index, th0->control(), th0->pulseId(), th0->time.value(), th0->env, th0->evtCounter,
                 th0->_opaque[0], th0->_opaque[1]);

  for (unsigned i = 1; i < m_pool.panels().size(); ++i) {
    bool ne = false;
    const auto dscN = (DmaDsc*)(m_pool.hostWrtBufsVec_h()[i][index]);
    const auto thN  = (TimingHeader*)&dscN[1];
    ne |= thN->control()    != th0->control();
    ne |= thN->pulseId()    != th0->pulseId();
    ne |= thN->time.value() != th0->time.value();
    ne |= thN->env          != th0->env;
    ne |= thN->evtCounter   != th0->evtCounter;

    if (ne) {
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
  const auto& hostWriteBufs = m_pool.hostWrtBufsVec_h()[0]; // When no error, hdrs in all are the same
  const uint32_t bufferMask = m_collectorQueue.h->size() - 1;

  unsigned head = m_collectorQueue.h->consume(); // This can block
  unsigned tail = m_last;
  while (tail != head) {
    const volatile auto dsc = (DmaDsc*)(hostWriteBufs[tail]);
    const volatile auto th  = (TimingHeader*)&dsc[1];

    uint64_t pid;
    while (!m_terminate_h.load(std::memory_order_acquire)) {
      pid = th->pulseId();
      if (pid > m_lastPid)  break;
      if (!m_lastPid && !pid)  break; // Expect lastPid to be 0 only on startup
    }
    if (m_terminate_h.load(std::memory_order_acquire))  break;
    if (!pid)  continue;              // Search for a DMA buffer with data in it
    m_lastPid = pid;

    // Handle the case when the headers don't match across panels
    // @todo: Too expensive?  This fetches the headers of all panels from the GPU
    //        Maybe do the test on the GPU and set a flag if they differ and
    //        print here when it is set
    unsigned dmas, ths;
    if ( (dmas = _checkDmaDsc(tail)) || (ths = _checkTimingHeader(tail)) ) {
      // Assume we can recover from non-matching panel headers
      logging::error("Headers differ between panels: DmaDsc: %02x, TimingHeader: %02x", dmas, ths);
      freeDma(tail);                    // Leaves event mask = 0
      metrics.m_nHdrMismatch += 1;
      continue;
    }

    uint32_t size = dsc->size;          // Size of the DMA
    uint32_t index = tail;
    uint32_t lane = 0;      // The lane is always 0 for GPU-enabled PGP devices
    metrics.m_dmaSize   = size;
    metrics.m_dmaBytes += size;

    // Check for DMA buffer overflow
    if (dsc->error & 0x4) {
      logging::critical("%d DMA overflowed buffer: %d vs %d", tail, size, m_pool.dmaSize());
      abort();                          // @todo: Still necessary to abort?
    }

    const Pds::TimingHeader* timingHeader = det->getTimingHeader(tail);

    // Measure TimingHeader arrival latency as early as possible
    if (timingHeader->pulseId() - m_latPid > 1300000/14) { // 10 Hz
        metrics.m_latency = Eb::latency<us_t>(timingHeader->time);
        m_latPid = timingHeader->pulseId();
    }
    if (timingHeader->error()) {
        if (metrics.m_nTmgHdrError < 5) { // Limit prints at rate
            logging::error("Timing header error bit is set");
        }
        metrics.m_nTmgHdrError += 1;
    }

    uint32_t evtCounter = timingHeader->evtCounter & EvtCtrMask;
    unsigned pgpIndex = evtCounter & bufferMask;
    PGPEvent*  event  = &m_pool.pgpEvents[pgpIndex];
    if (event->mask)  printf("*** PGPEvent mask != 0 for ctr %d\n", pgpIndex);
    DmaBuffer* buffer = &event->buffers[lane]; // @todo: Do we care about this?
    buffer->size = size;                       //   "
    buffer->index = index;                     //   "
    event->mask |= (1 << lane);

    m_pool.allocateDma(); // DMA buffer was allocated when f/w incremented evtCounter

    if (dsc->error) {
      // Assume we can recover from non-overflow DMA errors
      if (metrics.m_nDmaErrors < 5) {   // Limit prints at rate
        logging::error("DMA error 0x%x", dsc->error);
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
                   dsc->error);

    if (transitionId == TransitionId::BeginRun) {
      resetEventCounter();              // Compensate for the ClearReadout sent before BeginRun
    }
    if (evtCounter != ((m_lastComplete + 1) & EvtCtrMask)) {
      if (m_lastTid != TransitionId::Unconfigure) {
        if ((metrics.m_nPgpJumps < 5) || m_para.verbose) { // Limit prints at rate
          auto evtCntDiff = evtCounter - m_lastComplete;
          logging::error("%sPGPReader: Jump in TimingHeader evtCounter %u -> %u | difference %d, DMA size %u%s",
                         RED_ON, m_lastComplete, evtCounter, evtCntDiff, size, RED_OFF);
          logging::error("new data: %08x %08x %08x %08x %08x %08x  (%s)",
                         data[0], data[1], data[2], data[3], data[4], data[5], TransitionId::name(transitionId));
          logging::error("lastData: %08x %08x %08x %08x %08x %08x  (%s)",
                         m_lastData[0], m_lastData[1], m_lastData[2], m_lastData[3], m_lastData[4], m_lastData[5], TransitionId::name(m_lastTid));
        }
        handleBrokenEvent(*event);
        freeDma(event);                 // Leaves event mask = 0
        metrics.m_nPgpJumps += 1;
        continue;                       // Throw away out-of-sequence events
      } else if (transitionId != TransitionId::Configure) {
        freeDma(event);                 // Leaves event mask = 0
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
      if (missingRogs) {
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
  }
  unsigned nEvents = (head - m_last) & bufferMask;
  m_last = tail;

  return nEvents;
}
