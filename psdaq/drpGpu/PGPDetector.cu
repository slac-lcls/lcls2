#include "PGPDetector.hh"

#include "Detector.hh"
#include "Worker.hh"

#include "psalg/utils/SysLog.hh"
#include "psdaq/service/MetricExporter.hh"
#include "psdaq/service/EbDgram.hh"
#include "psdaq/eb/TebContributor.hh"
#include "drp/DrpBase.hh"

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Pds;
using namespace Drp;
using namespace Drp::Gpu;


PGPDetector::PGPDetector(const Parameters& para, DrpBase& drp, Gpu::Detector* det) :
  m_para       (para),
  m_drp        (drp),
  m_det        (det),
  m_terminate_h(false),
  m_nNoTrDgrams(0)
{
  printf("*** Gpu::PGPDetector 1\n");
  if (drp.pool.setMaskBytes(para.laneMask, m_det->virtChan)) {
    logging::critical("Failed to allocate lane/vc "
                      "- does another process have (one or more of) %s open?",
                      para.device.c_str());
    abort();
  }
  printf("*** Gpu::PGPDetector 2\n");

  // Set up thread termination flag in managed memory
  chkError(cudaMallocManaged(&m_terminate_d, sizeof(*m_terminate_d)));
  *m_terminate_d = 0;
  printf("*** Gpu::PGPDetector 3\n");

  // Retrieve the MemPoolGpu
  auto& pool = *drp.pool.getAs<MemPoolGpu>();
  printf("*** Gpu::PGPDetector 4\n");

  // Set up the communication queues between the various stages
  unsigned nBufs = pool.nbuffers();
  auto trgPrmtv = drp.triggerPrimitive();
  printf("*** Gpu::PGPDetector 5: trgPrmtv %p\n", trgPrmtv);
  printf("*** Gpu::PGPDetector 5: trgPrmtv sz %zu\n", trgPrmtv->size());
  auto trgPrmtvSz = trgPrmtv ? trgPrmtv->size() : 0;

  // Set up a Worker per PGP card (panel) to receive DMAed data and calibrate it
  chkError(cudaMalloc(&m_workerQueues_d, para.nworkers * sizeof(*m_workerQueues_d)));
  printf("*** Gpu::PGPDetector 5a\n");
  for (unsigned i = 0; i < para.nworkers; ++i) {
    // Set up buffer index allocator for DMA to Collector comms
    m_workerQueues_h.emplace_back(nBufs, drp.pool.dmaCount(), *m_terminate_d);
    printf("*** Gpu::PGPDetector 5b %d wq base %p\n", i, m_workerQueues_d);
    auto wq = &m_workerQueues_d[i];
    printf("*** Gpu::PGPDetector 5c %d, wq %p\n", i, wq);

    m_workers.emplace_back(i, para, pool, wq, *det, trgPrmtvSz, *m_terminate_d);
    printf("*** Gpu::PGPDetector 5d %d\n", i);
  }
  chkError(cudaMemcpy( m_workerQueues_d, m_workerQueues_h.data(), m_workerQueues_h.size() * sizeof(*m_workerQueues_d), cudaMemcpyHostToDevice));
  printf("*** Gpu::PGPDetector 6\n");

  // Set up buffer index allocator for Collector to Host comms
  m_collectorQueue.h = new Gpu::RingIndexDtoH(nBufs, m_terminate_h, *m_terminate_d);
  chkError(cudaMalloc(&m_collectorQueue.d,                     sizeof(*m_collectorQueue.d)));
  chkError(cudaMemcpy( m_collectorQueue.d, m_collectorQueue.h, sizeof(*m_collectorQueue.d), cudaMemcpyHostToDevice));
  printf("*** Gpu::PGPDetector 7\n");

  // Create the event building collector, which calculates the TEB input data
  // @todo: The TriggerPrimitive object in det is dynamically loaded to pick
  //        up the TEB input data creation algorithm, e.g., peak finder
  m_collector = std::make_unique<Collector>(para, pool, m_workerQueues_d, m_collectorQueue, trgPrmtv, m_terminate_h, *m_terminate_d);
  printf("*** Gpu::PGPDetector 8\n");

  // Set up buffer index allocator for Host to Reducer comms

  // Create the data reducer
  // @todo: Needs to be dynamically loaded to pick up the problem-specific
  //        reduction algorithm, e.g., SZ, angular integration, etc.
  // @todo: m_reducer = std::make_unique<Reducer>(para, pool, *det);

  // Create the data recorder
  // @todo: The recorder orchestrates getting the data written to a file and
  //        the forwarding of the data to the MEB
  // @todo: m_recorder = std::make_unique<Recorder>(para, pool, *det);
}

PGPDetector::~PGPDetector()
{
  printf("*** PGPDetector dtor 1\n");
  // Try to take things down gracefully when an exception takes us off the
  // normal path so that the most chance is given for prints to show up
  shutdown();
  printf("*** PGPDetector dtor 2\n");

  // @todo: m_recorder.reset();
  printf("*** PGPDetector dtor 3\n");

  // @todo: m_reducer.reset();
  printf("*** PGPDetector dtor 4\n");

  m_collector.reset();
  chkError(cudaFree(m_collectorQueue.d));
  delete m_collectorQueue.h;
  printf("*** PGPDetector dtor 5\n");

  m_workers.clear();
  //for (unsigned i = 0; i < m_para.nworkers; ++i) {
  //  chkError(cudaFree(m_workerQueues_d[i]));
  //}
  m_workerQueues_h.clear();
  chkError(cudaFree(m_workerQueues_d));
  printf("*** PGPDetector dtor 6\n");

  chkError(cudaFree(m_terminate_d));
  printf("*** PGPDetector dtor 7\n");
}

int PGPDetector::_setupMetrics(const std::shared_ptr<Pds::MetricExporter> exporter)
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
  //uint64_t nbuffers = m_para.nworkers * m_drp.pool.nbuffers();
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

void PGPDetector::collector(std::shared_ptr<MetricExporter> exporter)
{
  // Set up monitoring
  if (exporter) {
    int rc = _setupMetrics(exporter);
    if (rc)  return;
  }

  m_drp.pool.resetCounters();         // Avoid jumps in EbReceiver

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

  auto& tebContributor   = m_drp.tebContributor();
  auto  triggerPrimitive = m_drp.triggerPrimitive();

  const uint32_t bufferMask = m_drp.pool.nbuffers() - 1;
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
      PGPEvent* event = &m_drp.pool.pgpEvents[pgpIndex];
      if (event->mask == 0)
        continue;                       // Skip broken event

      auto pid = timingHeader->pulseId();
      if (pid <= lastPid)
        logging::error("PulseId did not advance: %014lx <= %014lx", pid, lastPid);
      lastPid = pid;

      // Allocate a pebble buffer
      event->pebbleIndex = m_drp.pool.allocate(); // This can block
      unsigned pebbleIndex = event->pebbleIndex;
      Src src = m_det->nodeId;
      TransitionId::Value transitionId = timingHeader->service();

      // Make a new dgram in the pebble
      // It must be an EbDgram in order to be able to send it to the MEB
      auto dgram = new(m_drp.pool.pebble[pebbleIndex]) EbDgram(*timingHeader, src, m_para.rogMask);

      // @todo: Temporary: Move to after when Reduce is done with calibData
      m_collector->freeDma(event);

      // Prepare the trigger primitive with whatever input is needed for the TEB to make trigger decisions
      auto l3InpBuf = tebContributor.fetch(pebbleIndex);
      auto l3InpDg  = new(l3InpBuf) EbDgram(*dgram);

      if (transitionId == TransitionId::L1Accept) {
        if (triggerPrimitive) { // else this DRP doesn't provide TEB input
          // Copy the TEB input data from the GPU into the TEB input datagram
          const void* l3BufEnd = (char*)l3InpDg + sizeof(*l3InpDg) + triggerPrimitive->size();
          auto buf = l3InpDg->xtc.alloc(triggerPrimitive->size(), l3BufEnd);
          memcpy(buf, &timingHeader[1], triggerPrimitive->size()); // @todo: cudaMemcpy() needed?
        }
      } else {
        logging::debug("PGPCollector saw %s @ %u.%09u (%014lx)",
                       TransitionId::name(transitionId),
                       dgram->time.seconds(), dgram->time.nanoseconds(), dgram->pulseId());

        // Allocate a transition datagram from the pool
        EbDgram* trDgram = m_drp.pool.allocateTr();
        if (!trDgram)  continue;        // Can occur when shutting down
        m_drp.pool.transitionDgrams[pebbleIndex] = trDgram;

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
      tebContributor.process(pebbleIndex);

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

  logging::info("Collector is exiting");
}

void PGPDetector::shutdown()
{
  logging::info("Shutting down GPU collector");

  m_terminate_h.store(true);
  m_terminate_d->store(1);

  // Flush the DMA buffers
  // @todo: flush();
}
