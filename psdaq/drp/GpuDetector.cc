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
    m_para(para),
    m_pool(pool),
    m_det(det),
    m_terminate(false)
{
    if (pool.setMaskBytes(para.laneMask, m_det->virtChan)) {
        logging::critical("Failed to allocate lane/vc "
                          "- does another process have %s open?", para.device.c_str());
        abort();
    }

    for (unsigned i = 0; i < para.nworkers; ++i) {
      //m_workers.emplace_back(i, para, pool);
        m_workers.push_back(new GpuWorker(i, para, pool)); // @todo: revisit
        m_workerQueues.emplace_back(SPSCQueue<Batch>(pool.nbuffers()));
    }
}

GpuDetector::~GpuDetector()
{
    // Try to take things down gracefully when an exception takes us off the
    // normal path so that the most chance is given for prints to show up
    shutdown();
}

void GpuDetector::collector(std::shared_ptr<MetricExporter> exporter,
                            Eb::TebContributor& tebContributor,
                            DrpBase& drp)
{
    uint64_t nevents = 0L;

    // setup monitoring
    std::map<std::string, std::string> labels{{"instrument", m_para.instrument},
                                              {"partition", std::to_string(m_para.partition)},
                                              {"detname", m_para.detName},
                                              {"alias", m_para.alias}};
    exporter->add("drp_event_rate", labels, MetricType::Rate,
                  [&](){return nevents;});

    auto queueLength = [](std::vector<SPSCQueue<Batch> >& vec)
        { size_t sum = 0;  for (auto& q: vec) sum += q.guess_size();  return sum; };
    uint64_t nbuffers = m_para.nworkers * m_pool.nbuffers();
    exporter->constant("drp_worker_queue_depth", labels, nbuffers);

    exporter->add("drp_worker_output_queue", labels, MetricType::Gauge,
                  [&](){return queueLength(m_workerQueues);});

    uint64_t nDmaRet = 0L;
    exporter->add("drp_num_dma_ret", labels, MetricType::Gauge,
                  [&](){return nDmaRet;});
    //exporter->add("drp_pgp_byte_rate", labels, MetricType::Rate,
    //              [&](){return m_workers[?].dmaBytes();});
    //exporter->add("drp_dma_size", labels, MetricType::Gauge,
    //              [&](){return m_workers[?].dmaSize();});
    //exporter->add("drp_th_latency", labels, MetricType::Gauge,
    //              [&](){return latency();});
    //exporter->add("drp_num_dma_errors", labels, MetricType::Gauge,
    //              [&](){return nDmaErrors();});
    //exporter->add("drp_num_no_common_rog", labels, MetricType::Gauge,
    //              [&](){return nNoComRoG();});
    //exporter->add("drp_num_missing_rogs", labels, MetricType::Gauge,
    //              [&](){return nMissingRoGs();});
    //exporter->add("drp_num_th_error", labels, MetricType::Gauge,
    //              [&](){return nTmgHdrError();});
    //exporter->add("drp_num_pgp_jump", labels, MetricType::Gauge,
    //              [&](){return nPgpJumps();});
    //exporter->add("drp_num_no_tr_dgram", labels, MetricType::Gauge,
    //              [&](){return nNoTrDgrams();});

    m_pool.resetCounters();             // Avoid jumps in EbReceiver

    for (auto& worker : m_workers) {
        // Start the GPU streams
        worker->start(m_workerQueues[worker->worker()], m_det);
    }

    logging::info("Collector is starting with process ID %lu\n", syscall(SYS_gettid));

    const uint32_t bufferMask = m_pool.nbuffers() - 1;
    auto triggerPrimitive = drp.triggerPrimitive(); // @todo: Revisit whether we want drp here
    Batch batch;
    unsigned worker = 0;
    bool rc = m_workerQueues[worker].pop(batch);
    logging::debug("Worker %d popped batch %u, size %zu\n", worker, batch.start, batch.size);
    while (rc) {
        for (unsigned i=0; i<batch.size; i++) {
            nevents++;

            // @todo: This needs to be significantly reworked to support event
            //        building the segments and then handling the combined event.
            //        As it is now, it will handle only one segment/worker.
            unsigned pgpIndex = (batch.start + i) & bufferMask;
            PGPEvent* event = &m_pool.pgpEvents[pgpIndex];
            if (event->mask == 0)
                continue;               // Skip broken event

            // The data payload for this dgram remains in the GPU
            // This dgram header is prepended to the payload in the GPU when the TEB result is
            // provided to the GPU and when it is determined that the dgram is to be recorded
            // @todo: This merging is done in the CPU if file writing isn't available from the GPU
            auto pebbleIndex = event->pebbleIndex;
            auto dgram = (EbDgram*)(m_pool.pebble[pebbleIndex]);

            // @todo: Okay to reuse event buffer after this point?
            m_workers[0]->freeDma(event);  // Doesn't matter which worker?

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
                    if (!trDgram)  continue; // Can occur when shutting down
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
                    if (!trDgram)  continue; // Can occur when shutting down
                    memcpy((void*)trDgram, (const void*)dgram, sizeof(*dgram) - sizeof(dgram->xtc));

                    // copy the temporary xtc created on phase 1 of the transition
                    // into the real location
                    Xtc& trXtc = m_det->transitionXtc();
                    trDgram->xtc = trXtc; // Preserve header info, but allocate to check fit
                    const void* bufEnd  = (char*)trDgram + m_para.maxTrSize;
                    auto payload = trDgram->xtc.alloc(trXtc.sizeofPayload(), bufEnd);
                    memcpy(payload, (const void*)trXtc.payload(), trXtc.sizeofPayload());

                    // Disable transitions terminate batches
                    assert(i == batch.size - 1);
                }
            }

            // Post level-3 input datagram to the TEB
            logging::debug("GpuCollector: Sending input %u to TEB", pebbleIndex);
            tebContributor.process(pebbleIndex);
        }
        worker = (worker + 1) % m_para.nworkers;

        // Time out batches for the TEB
        while (!m_workerQueues[worker].try_pop(batch)) { // Poll
            if (tebContributor.timeout()) {              // After batch is timed out,
                rc = m_workerQueues[worker].popW(batch); // pend
                break;
            }
        }
        logging::debug("Worker %d popped batch %u, size %zu\n", worker, batch.start, batch.size);
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
    for (unsigned i = 0; i < m_para.nworkers; i++) {
        m_workerQueues[i].shutdown();
    }

    // Flush the DMA buffers
    // @todo: flush();
}
