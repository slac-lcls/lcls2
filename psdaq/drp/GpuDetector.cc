#include "GpuDetector.hh"

#include "psdaq/aes-stream-drivers/DmaDriver.h"
#include "psdaq/service/MetricExporter.hh"
#include "psalg/utils/SysLog.hh"
#include "psdaq/eb/TebContributor.hh"
#include "DrpBase.hh"
#include "GpuWorker.hh"

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Pds;
using namespace Drp;


GpuDetector::GpuDetector(const Parameters& para, MemPoolGpu& pool, Detector* det) :
    m_para       (para),
    m_pool       (pool),
    m_det        (det),
    m_terminate  (false),
    m_nNoTrDgrams(0)
{
    if (pool.setMaskBytes(para.laneMask, m_det->virtChan)) {
        logging::critical("Failed to allocate lane/vc "
                          "- does another process have %s open?", para.device.c_str());
        abort();
    }

    for (unsigned i = 0; i < para.nworkers; ++i) {
      //m_workers.emplace_back(i, para, pool);
        m_workers.push_back(new GpuWorker(i, para, pool)); // @todo: revisit
    }
}

GpuDetector::~GpuDetector()
{
    // Try to take things down gracefully when an exception takes us off the
    // normal path so that the most chance is given for prints to show up
    shutdown();

    for (auto& worker : m_workers) {
      delete worker;
    }
}

int GpuDetector::_setupMetrics(const std::shared_ptr<Pds::MetricExporter> exporter)
{
    std::map<std::string, std::string> labels{{"instrument", m_para.instrument},
                                              {"partition", std::to_string(m_para.partition)},
                                              {"detname", m_para.detName},
                                              {"alias", m_para.alias}};
    m_metrics.m_nevents = 0L;
    exporter->add("drp_event_rate", labels, MetricType::Rate,
                  [&](){return m_metrics.m_nevents.load();});

    //auto queueLength = [](std::vector<SPSCQueue<Batch> >& vec)
    //    { size_t sum = 0;  for (auto& q: vec) sum += q.guess_size();  return sum; };
    //uint64_t nbuffers = m_para.nworkers * m_pool.nbuffers();
    //exporter->constant("drp_worker_queue_depth", labels, nbuffers);
    //
    //exporter->add("drp_worker_output_queue", labels, MetricType::Gauge,
    //              [&](){return queueLength(m_workerQueues);});

    m_metrics.m_nDmaRet = 0;
    exporter->add("drp_num_dma_ret", labels, MetricType::Gauge,
                  [&](){return m_metrics.m_nDmaRet.load();});
    m_metrics.m_dmaBytes = 0;
    exporter->add("drp_pgp_byte_rate", labels, MetricType::Rate,
                  [&](){return m_metrics.m_dmaBytes.load();});
    m_metrics.m_dmaSize = 0;
    exporter->add("drp_dma_size", labels, MetricType::Gauge,
                  [&](){return m_metrics.m_dmaSize.load();});
    //exporter->add("drp_th_latency", labels, MetricType::Gauge,
    //              [&](){return latency();});
    m_metrics.m_nDmaErrors = 0;
    exporter->add("drp_num_dma_errors", labels, MetricType::Gauge,
                  [&](){return m_metrics.m_nDmaErrors.load();});
    m_metrics.m_nNoComRoG = 0;
    exporter->add("drp_num_no_common_rog", labels, MetricType::Gauge,
                  [&](){return m_metrics.m_nNoComRoG.load();});
    m_metrics.m_nMissingRoGs = 0;
    exporter->add("drp_num_missing_rogs", labels, MetricType::Gauge,
                  [&](){return m_metrics.m_nMissingRoGs.load();});
    m_metrics.m_nTmgHdrError = 0;
    exporter->add("drp_num_th_error", labels, MetricType::Gauge,
                  [&](){return m_metrics.m_nTmgHdrError.load();});
    m_metrics.m_nPgpJumps = 0;
    exporter->add("drp_num_pgp_jump", labels, MetricType::Gauge,
                  [&](){return m_metrics.m_nPgpJumps.load();});
    m_nNoTrDgrams = 0;
    exporter->add("drp_num_no_tr_dgram", labels, MetricType::Gauge,
                  [&](){return m_nNoTrDgrams;});

    return 0;
}

void GpuDetector::collector(std::shared_ptr<MetricExporter> exporter,
                            Eb::TebContributor& tebContributor,
                            DrpBase& drp)
{
    // setup monitoring
    if (exporter) {
      int rc = _setupMetrics(exporter);
      if (rc)  return;
    }

    m_pool.resetCounters();             // Avoid jumps in EbReceiver

    for (auto& worker : m_workers) {
        // Start the GPU streams
        ///worker->start(m_workerQueues[worker->worker()], m_det, m_metrics);
        worker->start(m_det, m_metrics);
    }

    logging::info("Collector is starting with process ID %lu\n", syscall(SYS_gettid));

    ///const uint32_t bufferMask = m_pool.nDmaBuffers() - 1;
    auto triggerPrimitive = drp.triggerPrimitive(); // @todo: Revisit whether we want drp here
    //Batch batch;
    unsigned dmaIdx;
    //unsigned worker = 0;
    ///unsigned dmaIdx = 0;
    ///bool rc = m_workerQueues[worker].pop(batch);
    bool rc = true;
    ///logging::debug("Worker %d popped batch %u, size %zu\n", worker, batch.start, batch.size);
    //unsigned lastPblIndex = 0;
    uint64_t lastPid = 0;
    while (true) {
        TimingHeader* timingHeader;
        uint64_t pid = 0;
        ///for (unsigned i=0; i<batch.size; i++) {
        for (auto& worker : m_workers) {
            rc = worker->dmaQueue().pop(dmaIdx);
            //printf("*** rc %d worker %d dmaIdx %u\n", rc, worker->worker(), dmaIdx);
            if (!rc)  break;

            ///unsigned pgpIndex = (batch.start + i) & bufferMask;
            ///PGPEvent* event = &m_pool.pgpEvents[pgpIndex];
            ///if (event->mask == 0) {
            ///    printf("*** Worker %d skipping broken event %d\n", worker, batch.start+i);
            ///    continue;               // Skip broken event
            ///}

            // The data payload for this dgram remains in the GPU
            // This dgram header is prepended to the payload in the GPU when the TEB result is
            // provided to the GPU and when it is determined that the dgram is to be recorded
            // @todo: This merging is done in the CPU if file writing isn't available from the GPU
            ///auto pebbleIndex = event->pebbleIndex;
            ///auto dgram = (EbDgram*)(m_pool.pebble[pebbleIndex]);

            ///const unsigned lane = 0;
            ///DmaBuffer* buffer = &event->buffers[lane];
            ///const Pds::TimingHeader* timingHeader = m_workers[worker]->timingHeader(buffer->index);
            timingHeader = worker->timingHeader(dmaIdx);
            // @todo: Verify that the pulseId from each worker is the same
            if (!pid) {
              pid = timingHeader->pulseId();
              //printf("*** worker %d pid %014lx\n", worker->worker(), pid);
            }
            else if (timingHeader->pulseId() != pid)
              logging::error("Worker %d has a non-matching pulseId %014lx vs %014lx",
                             worker->worker(), timingHeader->pulseId(), pid);

            // @todo: stash TEB input from after TH
        }
        if (!rc)  break;

        if (pid <= lastPid)
            logging::error("PulseId did not advance: %014lx <= %014lx", pid, lastPid);
        lastPid = pid;

        // Allocate a pebble buffer
        auto counter       = m_pool.allocate(); // This can block  @todo: allocate should return pebbleIndex (already masked)?
        auto pebbleIndex   = counter & (m_pool.nbuffers() - 1);
        ///event->pebbleIndex = pebbleIndex;

        //if (pebbleIndex != ((lastPblIndex + dmaIdx) & (m_pool.nbuffers() - 1))) {
        //  printf("*** pblIdx %u, last %u, Worker %d, dmaIdx %d, nbufs %d, pid %014lx, %014lx\n",
        //         pebbleIndex, (lastPblIndex + dmaIdx) & (m_pool.nbuffers() - 1), m_worker, dmaIdx,
        //         m_pool.nbuffers(), pid, timingHeader->pulseId());
        //}
        //lastPblIndex += 4;

        // Make a new dgram in the pebble
        // It must be an EbDgram in order to be able to send it to the MEB
        if (!m_pool.pebble[pebbleIndex]) printf("*** pbl[%d] is NULL\n", pebbleIndex);
        auto dgram = new(m_pool.pebble[pebbleIndex]) EbDgram(*timingHeader, m_det->nodeId, m_para.rogMask);
        if (!dgram) printf("*** dgram %p at pblIdx %d\n", dgram, pebbleIndex);

        // @todo: Okay to reuse event buffer after this point?
        ///m_workers[worker]->freeDma(event);  // Doesn't matter which worker?
        for (auto& worker : m_workers)
            worker->freeDma(dmaIdx);  // @todo: Deal with different argument

        // Prepare the trigger primitive with whatever input is needed for the TEB to make trigger decisions
        auto l3InpBuf = tebContributor.fetch(pebbleIndex);
        auto l3InpDg  = new(l3InpBuf) EbDgram(*dgram);

        TransitionId::Value transitionId = dgram->service();
        if (transitionId == TransitionId::L1Accept) {
          if (triggerPrimitive) { // else this DRP doesn't provide input
            // @todo: Move this to GpuWorker?  tp->event() nominally
            //        analyzes the detector's data and extracts info to
            //        form the TEB input in l3InpDg->xtc
            const void* l3BufEnd = (char*)l3InpDg + sizeof(*l3InpDg) + triggerPrimitive->size();
            triggerPrimitive->event(m_pool, pebbleIndex, dgram->xtc, l3InpDg->xtc, l3BufEnd);
          }
        } else {
          // Allocate a transition datagram from the pool.  Since a
          // SPSCQueue is used (not an SPMC queue), this can be done here,
          // but not in the workers or there will be concurrency issues.
          m_pool.transitionDgrams[pebbleIndex] = m_pool.allocateTr();
          if (!m_pool.transitionDgrams[pebbleIndex]) {
            ++m_nNoTrDgrams;
            m_pool.freePebble(); // Avoid leaking pebbles on errors
            break;               // Can happen during shutdown
          }
          if (transitionId == TransitionId::SlowUpdate) {
            logging::debug("GpuCollector saw %s @ %u.%09u (%014lx)",
                           TransitionId::name(transitionId),
                           dgram->time.seconds(), dgram->time.nanoseconds(), dgram->pulseId());

            // Find the transition dgram in the pool and initialize its header
            EbDgram*    trDgram = m_pool.transitionDgrams[pebbleIndex];
            const void* bufEnd  = (char*)trDgram + m_para.maxTrSize;
            if (!trDgram) {
              printf("**s No trDgram for SlowUpdate %d\n", pebbleIndex); //batch.start+i);
              continue; // Can occur when shutting down
            }
            memcpy((void*)trDgram, (const void*)dgram, sizeof(*dgram) - sizeof(dgram->xtc));
            // @todo: This is mostly done on the GPU?
            m_det->slowupdate(trDgram->xtc, bufEnd);
          } else {                // Transition
            logging::debug("GpuCollector saw %s @ %u.%09u (%014lx)",
                           TransitionId::name(transitionId),
                           dgram->time.seconds(), dgram->time.nanoseconds(),
                           dgram->pulseId());

            // Initialize the transition dgram's header
            EbDgram* trDgram = m_pool.transitionDgrams[pebbleIndex];
            if (!trDgram) {
              printf("**t No trDgram for %s %d\n", TransitionId::name(transitionId), pebbleIndex); //batch.start+i);
              continue; // Can occur when shutting down
            }
            memcpy((void*)trDgram, (const void*)dgram, sizeof(*dgram) - sizeof(dgram->xtc));

            // copy the temporary xtc created on phase 1 of the transition
            // into the real location
            Xtc& trXtc = m_det->transitionXtc();
            trDgram->xtc = trXtc; // Preserve header info, but allocate to check fit
            const void* bufEnd  = (char*)trDgram + m_para.maxTrSize;
            auto payload = trDgram->xtc.alloc(trXtc.sizeofPayload(), bufEnd);
            memcpy(payload, (const void*)trXtc.payload(), trXtc.sizeofPayload());

            // Disable transitions terminate batches
            //assert(i == batch.size - 1);
          }
        }

        // Post level-3 input datagram to the TEB
        //printf("*** GpuCollector: Sending input %u to TEB", pebbleIndex);
        tebContributor.process(pebbleIndex);

        ///worker = (worker + 1) % m_para.nworkers;

        // Time out batches for the TEB
        /// while (!m_workerQueues[worker].try_pop(batch)) { // Poll
        ///     if (tebContributor.timeout()) {              // After batch is timed out,
        ///         rc = m_workerQueues[worker].popW(batch); // pend
        ///         break;
        ///     }
        /// }
        /// logging::debug("Worker %d popped batch %u, size %zu\n", worker, batch.start, batch.size);
    }

    for (auto& worker : m_workers) {
        // Stop the GPU streams
        worker->stop();
    }

    logging::info("Collector is exiting");
}

void GpuDetector::shutdown()
{
    if (m_terminate.load(std::memory_order_relaxed))
        return;                         // Already shut down

    logging::info("Shutting down GPU collector");
    m_terminate.store(true, std::memory_order_release);

    for (auto& worker : m_workers) {
        worker->dmaQueue().shutdown();
    }

    // Flush the DMA buffers
    // @todo: flush();
}
