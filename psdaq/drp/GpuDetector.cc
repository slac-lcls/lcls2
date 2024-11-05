#include "GpuDetector.hh"

#include "psdaq/service/MetricExporter.hh"
#include "psalg/utils/SysLog.hh"
#include "psdaq/eb/TebContributor.hh"
#include "DrpBase.hh"
#include <DmaDriver.h>
#include "GpuWorker.hh"

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Pds;
using namespace Drp;


GpuDetector::GpuDetector(const Parameters& para, DrpBase& drp, Detector* det) :
    PgpReader(para, drp.pool, MAX_RET_CNT_C, para.batchSize),
    m_drp(drp),
    m_det(det),
    m_collectorCpuQueue(drp.pool.nbuffers()),
    m_collectorGpuQueue(drp.pool.nbuffers()),
    m_terminate(false)
{
    if (drp.pool.setMaskBytes(para.laneMask, m_det->virtChan)) {
        logging::critical("Failed to allocate lane/vc "
                          "- does another process have %s open?", para.device.c_str());
        abort();
    }

    // Start the GPU streams
    m_det->gpuWorker()->start(m_collectorGpuQueue);
}

GpuDetector::~GpuDetector()
{
    // Stop the GPU streams
    m_det->gpuWorker()->stop();

    // Try to take things down gracefully when an exception takes us off the
    // normal path so that the most chance is given for prints to show up
    shutdown();
}

void GpuDetector::reader(std::shared_ptr<MetricExporter> exporter,
                         Eb::TebContributor& tebContributor)
{
    uint64_t nevents = 0L;
    const unsigned bufferMask = m_pool.nDmaBuffers() - 1;

    // Select timing messages to be DMAed to the CPU
    m_det->gpuWorker()->dmaMode(GpuWorker::CPU);

    // setup monitoring
    std::map<std::string, std::string> labels{{"instrument", m_para.instrument},
                                              {"partition", std::to_string(m_para.partition)},
                                              {"detname", m_para.detName},
                                              {"alias", m_para.alias}};
    exporter->add("drp_event_rate", labels, MetricType::Rate,
                  [&](){return nevents;});

    // @todo: Revisit
    //auto queueLength = [](std::vector<SPSCQueue<Batch> >& vec)
    //    { size_t sum = 0;  for (auto& q: vec) sum += q.guess_size();  return sum; };
    //uint64_t nbuffers = m_para.nworkers * m_pool.nbuffers();
    //exporter->constant("drp_worker_queue_depth", labels, nbuffers);
    //
    //exporter->add("drp_worker_input_queue", labels, MetricType::Gauge,
    //              [&](){return queueLength(m_workerInputQueues);});
    //
    //exporter->add("drp_worker_output_queue", labels, MetricType::Gauge,
    //              [&](){return queueLength(m_workerOutputQueues);});

    uint64_t nDmaRet = 0L;
    exporter->add("drp_num_dma_ret", labels, MetricType::Gauge,
                  [&](){return nDmaRet;});
    exporter->add("drp_pgp_byte_rate", labels, MetricType::Rate,
                  [&](){return dmaBytes();});
    exporter->add("drp_dma_size", labels, MetricType::Gauge,
                  [&](){return dmaSize();});
    exporter->add("drp_th_latency", labels, MetricType::Gauge,
                  [&](){return latency();});
    exporter->add("drp_num_dma_errors", labels, MetricType::Gauge,
                  [&](){return nDmaErrors();});
    exporter->add("drp_num_no_common_rog", labels, MetricType::Gauge,
                  [&](){return nNoComRoG();});
    exporter->add("drp_num_missing_rogs", labels, MetricType::Gauge,
                  [&](){return nMissingRoGs();});
    exporter->add("drp_num_th_error", labels, MetricType::Gauge,
                  [&](){return nTmgHdrError();});
    exporter->add("drp_num_pgp_jump", labels, MetricType::Gauge,
                  [&](){return nPgpJumps();});
    exporter->add("drp_num_no_tr_dgram", labels, MetricType::Gauge,
                  [&](){return nNoTrDgrams();});

    resetEventCounter();

    logging::info("GPU reader is starting with process ID %lu", syscall(SYS_gettid));

    while (1) {
        if (m_terminate.load(std::memory_order_relaxed)) {
            break;
        }
        int32_t ret = read();
        nDmaRet = ret;

        for (int b=0; b < ret; b++) {
            const TimingHeader* timingHeader = handle(m_det, b);
            if (!timingHeader)  continue;

            nevents++;

            TransitionId::Value transitionId = timingHeader->service();

            bool stateTransition = (transitionId != TransitionId::L1Accept) &&
                                   (transitionId != TransitionId::SlowUpdate);

            uint32_t pgpIndex = timingHeader->evtCounter & bufferMask;

            PGPEvent* event = &m_pool.pgpEvents[pgpIndex];
            if (event->mask == 0)
                continue;               // Skip broken event

            unsigned pebbleIndex = event->pebbleIndex;
            Src src = m_det->nodeId;
            EbDgram* dgram = new(m_pool.pebble[pebbleIndex]) EbDgram(*timingHeader,
                                                                     src, m_para.rogMask);

            if (stateTransition) {
                logging::debug("GpuDetector saw %s @ %u.%09u (%014lx)",
                               TransitionId::name(transitionId),
                               dgram->time.seconds(), dgram->time.nanoseconds(),
                               dgram->pulseId());

                // Initialize the transition dgram's header
                EbDgram* trDgram = m_pool.transitionDgrams[pebbleIndex];
                if (!trDgram)  continue; // Can occur when shutting down
                memcpy((void*)trDgram, (const void*)dgram, sizeof(*dgram) - sizeof(dgram->xtc));

                // copy the temporary xtc created on phase 1 of the transition
                // into the real location
                Xtc& trXtc = m_det->transitionXtc();
                trDgram->xtc = trXtc; // Preserve header info, but allocate to check fit
                const void* bufEnd  = (char*)trDgram + m_para.maxTrSize;
                auto payload = trDgram->xtc.alloc(trXtc.sizeofPayload(), bufEnd);
                memcpy(payload, (const void*)trXtc.payload(), trXtc.sizeofPayload());

                // Prepare the trigger primitive with whatever input is needed for the TEB to make trigger decisions
                new(tebContributor.fetch(pebbleIndex)) EbDgram(*dgram);

                // This must be done before Collector can process the Enable to
                // ensure that the next TimingHeader goes to the GPU and not the CPU
                if (transitionId == TransitionId::Enable) {
                    // Throw "switch" to have subsequent timing headers DMAed to the GPU
                    m_det->gpuWorker()->dmaMode(GpuWorker::GPU); // Bypass CPU

                    // Update the GPU with the next DMA index to expect
                    m_det->gpuWorker()->dmaIndex(pgpIndex);
                }

                // Queue transitions to Collector in order to maintain coherency
                m_collectorCpuQueue.push(pgpIndex);
            } else {                    // L1Accepts and SlowUpdates
                logging::critical("GpuDetector unexpectedly saw %s @ %u.%09u (%014lx)",
                                  TransitionId::name(transitionId),
                                  dgram->time.seconds(), dgram->time.nanoseconds(),
                                  dgram->pulseId());
                abort();
            }
        }
    }

    logging::info("PGP reader is exiting");
}

void GpuDetector::collector(Eb::TebContributor& tebContributor)
{
    logging::info("Collector is starting with process ID %lu\n", syscall(SYS_gettid));

    unsigned pgpIndex;
    while (m_collectorCpuQueue.pop(pgpIndex)) {
        auto event = &m_pool.pgpEvents[pgpIndex];
        auto pebbleIndex = event->pebbleIndex;
        auto l3InpDg = static_cast<EbDgram*>(tebContributor.fetch(pebbleIndex));
        freeDma(event);                  // Release DMA buffer - @todo: Do in reader thread?
        tebContributor.process(l3InpDg); // Queue input dgram to TEB

        // Get transitionId from the TEB's input dgram
        if (l3InpDg->service() == TransitionId::Enable) {
            // This doesn't return until Disable is seen, processing batches from the GPU
            _gpuCollector(tebContributor);
        }
    }

    logging::info("Collector is exiting");
}

void GpuDetector::_gpuCollector(Eb::TebContributor& tebContributor)
{
    const uint32_t bufferMask = m_pool.nbuffers() - 1;
    auto triggerPrimitive = m_drp.triggerPrimitive();
    bool sawDisable = false;
    Batch batch;
    bool rc = m_collectorGpuQueue.pop(batch);
    while (rc) {
        for (unsigned i=0; i<batch.size; i++) {
            unsigned pgpIndex = (batch.start + i) & bufferMask;
            PGPEvent* event = &m_pool.pgpEvents[pgpIndex];
            if (event->mask == 0)
                continue;               // Skip broken event

            // The data payload for this dgram remains in the GPU
            // This dgram header is prepended to the payload in the GPU when the TEB result is
            // provided to the GPU and the it is determined that the dgram is to be recorded
            // @todo: This merging is done in the CPU if file writing isn't available from the GPU
            auto pebbleIndex = event->pebbleIndex;
            auto dgram = (EbDgram*)(m_pool.pebble[pebbleIndex]);

            // Prepare the trigger primitive with whatever input is needed for the TEB to make trigger decisions
            auto l3InpBuf = tebContributor.fetch(pebbleIndex);
            auto l3InpDg  = new(l3InpBuf) EbDgram(*dgram);

            TransitionId::Value transitionId = dgram->service();

            if (transitionId == TransitionId::L1Accept) {
                if (triggerPrimitive) { // else this DRP doesn't provide input
                    // @todo: Move this to GpuWorker_impl?  tp->event() nominally analyzes the
                    //        detector's data and extracts info to form the TEB input in l3InpDg->xtc
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
                    if (!trDgram)  continue; // Can occur when shutting down
                    memcpy((void*)trDgram, (const void*)dgram, sizeof(*dgram) - sizeof(dgram->xtc));
                    // @todo: This is mostly done on the GPU?
                    m_det->slowupdate(trDgram->xtc, bufEnd);
                } else if (transitionId == TransitionId::Disable) {
                    logging::debug("GpuCollector saw %s @ %u.%09u (%014lx)",
                                   TransitionId::name(transitionId),
                                   dgram->time.seconds(), dgram->time.nanoseconds(),
                                   dgram->pulseId());
                    sawDisable = true;

                    // Ensure PgpReader::handle() doesn't complain about evtCounter jumps
                    m_lastComplete = m_det->gpuWorker()->lastEvtCtr();

                    // Initialize the transition dgram's header
                    EbDgram* trDgram = m_pool.transitionDgrams[pebbleIndex];
                    if (!trDgram)  continue; // Can occur when shutting down
                    memcpy((void*)trDgram, (const void*)dgram, sizeof(*dgram) - sizeof(dgram->xtc));

                    // copy the temporary xtc created on phase 1 of the transition
                    // into the real location
                    // @todo: Is there any payload for a Disable?
                    Xtc& trXtc = m_det->transitionXtc();
                    trDgram->xtc = trXtc; // Preserve header info, but allocate to check fit
                    const void* bufEnd  = (char*)trDgram + m_para.maxTrSize;
                    auto payload = trDgram->xtc.alloc(trXtc.sizeofPayload(), bufEnd);
                    memcpy(payload, (const void*)trXtc.payload(), trXtc.sizeofPayload());

                    // Disable transitions terminate batches
                    assert(i == batch.size - 1);
                } else {
                    // The GPU should not have seen any other transitions
                    logging::error("GpuDetector unexpectedly saw %s @ %u.%09u (%014lx)",
                                   TransitionId::name(transitionId),
                                   dgram->time.seconds(), dgram->time.nanoseconds(),
                                   dgram->pulseId());
                    // @todo: abort(); ?
                }
            }

            // Post level-3 input datagram to the TEB
            logging::debug("GpuCollector: Pushing pbl %u to TEB\n", pebbleIndex);
            tebContributor.process(pebbleIndex);
        }

        // If Disable was seen, no more batches are expected from the GPU
        // Return to handling transitions by the CPU
        if (sawDisable)  {
            // Throw "switch" to have subsequent timing headers DMAed to the CPU
            m_det->gpuWorker()->dmaMode(GpuWorker::CPU); // Bypass GPU

            break;
        }

        // Time out batches for the TEB
        while (!m_collectorGpuQueue.try_pop(batch)) { // Poll
            if (tebContributor.timeout()) {           // After batch is timed out
                rc = m_collectorGpuQueue.popW(batch); // pend
                break;
            }
        }
    }
    logging::debug("GpuCollector returning to process transitions\n");
}

void GpuDetector::handleBrokenEvent(const PGPEvent&)
{
    // Nothing to do
}

void GpuDetector::resetEventCounter()
{
    PgpReader::resetEventCounter();
}

void GpuDetector::shutdown()
{
    if (m_terminate.load(std::memory_order_relaxed))
        return;                         // Already shut down

    logging::info("Shutting down GPU reader and collector");
    m_terminate.store(true, std::memory_order_release);
    auto nCpu = m_collectorCpuQueue.guess_size();
    auto nGpu = m_collectorGpuQueue.guess_size();
    if (nCpu)  logging::warning("Non-empty %s collector queue: %zu entries",
                                "CPU", nCpu);
    if (nGpu)  logging::warning("Non-empty %s collector queue: %zu entries",
                                "GPU", nGpu);
    m_collectorCpuQueue.shutdown();
    m_collectorGpuQueue.shutdown();

    // Flush the DMA buffers
    flush();
}
