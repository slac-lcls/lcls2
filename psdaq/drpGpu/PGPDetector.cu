#include "PGPDetector.hh"

#include "Detector.hh"
#include "Worker.hh"

#include "psalg/utils/SysLog.hh"
#include "psdaq/service/MetricExporter.hh"
#include "psdaq/service/EbDgram.hh"
#include "psdaq/eb/TebContributor.hh"
#include "drp/DrpBase.hh"

using json    = nlohmann::json;
using logging = psalg::SysLog;
using namespace XtcData;
using namespace Pds;
using namespace Drp;
using namespace Drp::Gpu;


PGPDrp::PGPDrp(Parameters&    parameters,
               MemPoolGpu&    memPool,
               Gpu::Detector* detector,
               ZmqContext&    context) :
  DrpBase(parameters, memPool, context),
  m_para       (parameters),
  m_det        (detector),
  m_terminate_h(false),
  m_nNoTrDgrams(0)
{
  if (pool.setMaskBytes(m_para.laneMask, m_det->virtChan)) {
    logging::critical("Failed to allocate lane/vc "
                      "- does another process have (one or more of) %s open?",
                      m_para.device.c_str());
    abort();
  }

  // Set up thread termination flag in managed memory
  chkError(cudaMallocManaged(&m_terminate_d, sizeof(*m_terminate_d)));
  *m_terminate_d = 0;
}

PGPDrp::~PGPDrp()
{
  chkError(cudaFree(m_terminate_d));
}

std::string PGPDrp::configure(const json& msg)
{
  std::string errorMsg = DrpBase::configure(msg);
  if (!errorMsg.empty()) {
    return errorMsg;
  }

  m_terminate_h.store(false, std::memory_order_release);
  m_terminate_d->store(0, cuda::memory_order_release);

  // Set up the communication queues between the various stages
  unsigned nBuffers = pool.nbuffers();
  auto trgPrimitive = triggerPrimitive();
  auto tpSz = trgPrimitive ? trgPrimitive->size() : 0;

  // Set up a Worker per PGP card (panel) to receive DMAed data and calibrate it
  chkError(cudaMalloc(&m_workerQueues_d, m_para.nworkers * sizeof(*m_workerQueues_d)));
  for (unsigned i = 0; i < m_para.nworkers; ++i) {
    // Set up buffer index allocator for DMA to Collector comms
    m_workerQueues_h.emplace_back(nBuffers, pool.dmaCount(), *m_terminate_d);
    auto wq = &m_workerQueues_d[i];

    m_workers.emplace_back(i, m_para, *pool.getAs<MemPoolGpu>(), wq, *m_det, tpSz, *m_terminate_d);
  }
  chkError(cudaMemcpy(m_workerQueues_d, m_workerQueues_h.data(), m_workerQueues_h.size() * sizeof(*m_workerQueues_d), cudaMemcpyHostToDevice));

  // Set up buffer index allocator for Collector to Host comms
  m_collectorQueue.h = new Gpu::RingIndexDtoH(nBuffers, m_terminate_h, *m_terminate_d);
  chkError(cudaMalloc(&m_collectorQueue.d,                     sizeof(*m_collectorQueue.d)));
  chkError(cudaMemcpy( m_collectorQueue.d, m_collectorQueue.h, sizeof(*m_collectorQueue.d), cudaMemcpyHostToDevice));

  // Create the event building collector, which calculates the TEB input data
  // @todo: The TriggerPrimitive object in det is dynamically loaded to pick
  //        up the TEB input data creation algorithm, e.g., peak finder
  m_collector = std::make_unique<Collector>(m_para, *pool.getAs<MemPoolGpu>(), m_workerQueues_d, m_collectorQueue, trgPrimitive, m_terminate_h, *m_terminate_d);

  // Set up buffer index allocator for Host to Reducer comms

  // Create the data reducer
  // @todo: Needs to be dynamically loaded to pick up the problem-specific
  //        reduction algorithm, e.g., SZ, angular integration, etc.
  // @todo: m_reducer = std::make_unique<Reducer>(para, pool, *det);

  // Create the data recorder
  // @todo: The recorder orchestrates getting the data written to a file and
  //        the forwarding of the data to the MEB
  // @todo: m_recorder = std::make_unique<Recorder>(para, pool, *det);

  m_collectorThread = std::thread(&PGPDrp::collector, std::ref(*this));

  return std::string();
}

unsigned PGPDrp::unconfigure()
{
  DrpBase::unconfigure(); // TebContributor must be shut down before the worker

  logging::info("Shutting down");

  // @todo: Right place for this?
  m_terminate_h.store(true, std::memory_order_release);
  m_terminate_d->store(1, cuda::memory_order_release);

  if (m_collectorThread.joinable()) {
    m_collectorThread.join();
    logging::info("Collector thread finished");
  }

  // @todo: m_recorder.reset();

  // @todo: m_reducer.reset();

  m_collector.reset();
  chkError(cudaFree(m_collectorQueue.d));
  delete m_collectorQueue.h;

  m_workers.clear();
  //for (unsigned i = 0; i < m_para.nworkers; ++i) {
  //  chkError(cudaFree(m_workerQueues_d[i]));
  //}
  m_workerQueues_h.clear();
  chkError(cudaFree(m_workerQueues_d));

  return 0;
}

int PGPDrp::_setupMetrics(const std::shared_ptr<Pds::MetricExporter> exporter)
{
  std::map<std::string, std::string> labels{{"instrument", m_para.instrument},
                                            {"partition", std::to_string(m_para.partition)},
                                            {"detname", m_para.detName},
                                            {"alias", m_para.alias}};
  m_colMetrics.m_nevents = 0L;
  exporter->add("drp_event_rate", labels, MetricType::Rate,
                [&](){return m_colMetrics.m_nevents.load();});

  //auto queueLength = [](std::vector<SPSCQueue<Batch> >& vec)
  //    { size_t sum = 0;  for (auto& q: vec) sum += q.guess_size();  return sum; };
  //uint64_t nbuffers = m_para.nworkers * pool.nbuffers();
  //exporter->constant("drp_worker_queue_depth", labels, nbuffers);
  //
  //exporter->add("drp_worker_output_queue", labels, MetricType::Gauge,
  //              [&](){return queueLength(m_workerQueues);});

  m_colMetrics.m_nDmaRet = 0;
  exporter->add("drp_num_dma_ret", labels, MetricType::Gauge,
                [&](){return m_colMetrics.m_nDmaRet.load();});
  m_colMetrics.m_dmaBytes = 0;
  exporter->add("drp_pgp_byte_rate", labels, MetricType::Rate,
                [&](){return m_colMetrics.m_dmaBytes.load();});
  m_colMetrics.m_dmaSize = 0;
  exporter->add("drp_dma_size", labels, MetricType::Gauge,
                [&](){return m_colMetrics.m_dmaSize.load();});
  //exporter->add("drp_th_latency", labels, MetricType::Gauge,
  //              [&](){return latency();});
  m_colMetrics.m_nDmaErrors = 0;
  exporter->add("drp_num_dma_errors", labels, MetricType::Gauge,
                [&](){return m_colMetrics.m_nDmaErrors.load();});
  m_colMetrics.m_nNoComRoG = 0;
  exporter->add("drp_num_no_common_rog", labels, MetricType::Gauge,
                [&](){return m_colMetrics.m_nNoComRoG.load();});
  m_colMetrics.m_nMissingRoGs = 0;
  exporter->add("drp_num_missing_rogs", labels, MetricType::Gauge,
                [&](){return m_colMetrics.m_nMissingRoGs.load();});
  m_colMetrics.m_nTmgHdrError = 0;
  exporter->add("drp_num_th_error", labels, MetricType::Gauge,
                [&](){return m_colMetrics.m_nTmgHdrError.load();});
  m_colMetrics.m_nPgpJumps = 0;
  exporter->add("drp_num_pgp_jump", labels, MetricType::Gauge,
                [&](){return m_colMetrics.m_nPgpJumps.load();});
  m_nNoTrDgrams = 0;
  exporter->add("drp_num_no_tr_dgram", labels, MetricType::Gauge,
                [&](){return m_nNoTrDgrams;});

  return 0;
}

void PGPDrp::collector()
{
  pool.resetCounters();                 // Avoid jumps in EbReceiver

  // Set up monitoring
  auto exporter = std::make_shared<MetricExporter>();
  if (exposer()) {
    exposer()->RegisterCollectable(exporter);

    if (_setupMetrics(exporter))  return;
  }

  // Start the Data Recorder
  // @todo: m_recorder->start();

  // Start the Data Reducer
  // @todo: m_reducer->start();

  // Start the Collector
  m_collector->start();

  // Start the GPU workers
  for (auto& worker : m_workers) {
    worker.start();
  }

  logging::info("Collector is starting with process ID %lu\n", syscall(SYS_gettid));

  auto trgPrimitive = triggerPrimitive();

  const uint32_t bufferMask = pool.nbuffers() - 1;
  uint64_t lastPid = 0;
  unsigned bufIndex = 0;                // Intermediate buffer index
  while (true) {
    if (m_terminate_h.load(std::memory_order_relaxed)) {
      break;
    }
    TimingHeader* timingHeader;
    auto nRet = m_collector->receive(m_det, m_colMetrics); // This can block
    m_colMetrics.m_nDmaRet.store(nRet);

    for (unsigned b = 0; b < nRet; ++b) {
      timingHeader = m_det->getTimingHeader(bufIndex);
      uint32_t pgpIndex = timingHeader->evtCounter & bufferMask;
      PGPEvent* event = &pool.pgpEvents[pgpIndex];
      if (event->mask == 0)
        continue;                       // Skip broken event

      auto pid = timingHeader->pulseId();
      if (pid <= lastPid)
        logging::error("PulseId did not advance: %014lx <= %014lx", pid, lastPid);
      lastPid = pid;

      // Allocate a pebble buffer
      event->pebbleIndex = pool.allocate(); // This can block
      unsigned pebbleIndex = event->pebbleIndex;
      Src src = m_det->nodeId;
      TransitionId::Value transitionId = timingHeader->service();

      // Make a new dgram in the pebble
      // It must be an EbDgram in order to be able to send it to the MEB
      auto dgram = new(pool.pebble[pebbleIndex]) EbDgram(*timingHeader, src, m_para.rogMask);

      // @todo: Temporary: Move to after when Reduce is done with calibData
      m_collector->freeDma(event);

      // Prepare the trigger primitive with whatever input is needed for the TEB to make trigger decisions
      auto l3InpBuf = tebContributor().fetch(pebbleIndex);
      auto l3InpDg  = new(l3InpBuf) EbDgram(*dgram);

      if (transitionId == TransitionId::L1Accept) {
        if (triggerPrimitive()) { // else this DRP doesn't provide TEB input
          // Copy the TEB input data from the GPU into the TEB input datagram
          auto tpSz = triggerPrimitive()->size();
          const void* l3BufEnd = (char*)l3InpDg + sizeof(*l3InpDg) + tpSz;
          auto buf = l3InpDg->xtc.alloc(tpSz, l3BufEnd);
          memcpy(buf, &timingHeader[1], tpSz); // @todo: cudaMemcpy() needed?
        }
      } else {
        logging::debug("PGPCollector saw %s @ %u.%09u (%014lx)",
                       TransitionId::name(transitionId),
                       dgram->time.seconds(), dgram->time.nanoseconds(), dgram->pulseId());

        // Allocate a transition datagram from the pool
        EbDgram* trDgram = pool.allocateTr();
        if (!trDgram)  continue;        // Can occur when shutting down
        pool.transitionDgrams[pebbleIndex] = trDgram;

        // Initialize the transition dgram's header
        memcpy((void*)trDgram, dgram, sizeof(*dgram) - sizeof(dgram->xtc));

        if (transitionId == TransitionId::SlowUpdate) {
          // Store the SlowUpdate's payload in the transition datagram
          const void* bufEnd  = (char*)trDgram + m_para.maxTrSize;
          m_det->slowupdate(trDgram->xtc, bufEnd); // @todo: Is this needed or done on the GPU?
        } else {                // Transition
          // copy the temporary xtc created on phase 1 of the transition
          // into the real location
          Xtc& trXtc = m_det->transitionXtc();
          trDgram->xtc = trXtc; // Preserve header info, but allocate to check fit
          const void* bufEnd  = (char*)trDgram + m_para.maxTrSize;
          auto payload = trDgram->xtc.alloc(trXtc.sizeofPayload(), bufEnd);
          memcpy(payload, (const void*)trXtc.payload(), trXtc.sizeofPayload());
        }
      }

      // Post level-3 input datagram to the TEB
      //printf("*** GpuCollector: Sending input %u to TEB", pebbleIndex);
      tebContributor().process(pebbleIndex);

      // Time out batches for the TEB
      /// while (!m_workerQueues[worker].try_pop(batch)) { // Poll
      ///     if (tebContributor.timeout()) {              // After batch is timed out,
      ///         rc = m_workerQueues[worker].popW(batch); // pend
      ///         break;
      ///     }
      /// }
      /// logging::debug("Worker %d popped batch %u, size %zu\n", worker, batch.start, batch.size);

      bufIndex = (bufIndex + 1) & bufferMask;
    }
  }

  // Flush the buffers
  // @todo: dmaFlush();
  pool.flushPebble();

  if (exposer())  exporter.reset();

  logging::info("Collector is exiting");
}
