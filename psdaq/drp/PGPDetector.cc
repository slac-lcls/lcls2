#include <iostream>
#include <limits.h>
#include "DataDriver.h"
#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "psdaq/service/Collection.hh"
#include "psdaq/service/MetricExporter.hh"
#include "psalg/utils/SysLog.hh"
#include "psdaq/eb/TebContributor.hh"
#include "psdaq/service/fast_monotonic_clock.hh"
#include "DrpBase.hh"
#include "PGPDetector.hh"
#include "EventBatcher.hh"

#ifndef POSIX_TIME_AT_EPICS_EPOCH
#define POSIX_TIME_AT_EPICS_EPOCH 631152000u
#endif

using logging = psalg::SysLog;
using ms_t = std::chrono::milliseconds;

using namespace Drp;

void workerFunc(const Parameters& para, DrpBase& drp, Detector* det,
                SPSCQueue<Batch>& inputQueue, SPSCQueue<Batch>& outputQueue)
{
    Batch batch;
    MemPool& pool = drp.pool;
    const unsigned bufferMask = pool.nDmaBuffers() - 1;
    auto& tebContributor = drp.tebContributor();
    auto triggerPrimitive = drp.triggerPrimitive();
    auto& tebPrms = drp.tebPrms();

    while (true) {
        if (!inputQueue.pop(batch)) {
            break;
        }

        for (unsigned i=0; i<batch.size; i++) {
            unsigned index = (batch.start + i) & bufferMask;
            PGPEvent* event = &pool.pgpEvents[index];
            if (event->mask == 0)
                continue;               // Skip broken event

            // get transitionId from the first lane in the event
            int lane = __builtin_ffs(event->mask) - 1;
            uint32_t dmaIndex = event->buffers[lane].index;
            const Pds::TimingHeader* timingHeader = det->getTimingHeader(dmaIndex);

            // make new dgram in the pebble
            // It must be an EbDgram in order to be able to send it to the MEB
            unsigned pebbleIndex = event->pebbleIndex;
            XtcData::Src src = det->nodeId;
            Pds::EbDgram* dgram = new(pool.pebble[pebbleIndex]) Pds::EbDgram(*timingHeader, src, para.rogMask);
            XtcData::TransitionId::Value transitionId = dgram->service();

            // Event
            if (transitionId == XtcData::TransitionId::L1Accept) {
                const void* bufEnd = (char*)dgram + pool.bufferSize();
                det->event(*dgram, bufEnd, event);

                // make sure the detector hasn't made the event too big
                if (dgram->xtc.extent > pool.bufferSize()) {
                    logging::critical("L1Accept: buffer size (%d) too small for requested extent (%d)",
                                      pool.bufferSize(), dgram->xtc.extent);
                    throw "Buffer too small";
                }

                // Prepare the trigger primitive with whatever input is needed for the TEB to meke trigger decisions
                auto l3InpBuf = tebContributor.fetch(pebbleIndex);
                Pds::EbDgram* l3InpDg = new(l3InpBuf) Pds::EbDgram(*dgram);
                if (triggerPrimitive) { // else this DRP doesn't provide input
                    const void* l3BufEnd = (char*)l3InpDg + sizeof(*l3InpDg) + triggerPrimitive->size();
                    triggerPrimitive->event(pool, pebbleIndex, dgram->xtc, l3InpDg->xtc, l3BufEnd);
                    size_t size = sizeof(*l3InpDg) + l3InpDg->xtc.sizeofPayload();
                    if (size > tebPrms.maxInputSize) {
                        logging::critical("L3 Input Dgram of size %zd overflowed buffer of size %zd", size, tebPrms.maxInputSize);
                        throw "Input Dgram overflowed buffer";
                    }
                }
            }
            // transitions
            else {
                logging::debug("PGPDetector saw %s @ %u.%09u (%014lx)",
                               XtcData::TransitionId::name(transitionId),
                               dgram->time.seconds(), dgram->time.nanoseconds(), dgram->pulseId());
                // Find the transition dgram in the pool and initialize its header
                Pds::EbDgram* trDgram = pool.transitionDgrams[pebbleIndex];
                const void*   bufEnd  = (char*)trDgram + para.maxTrSize;
                if (!trDgram)  continue; // Can occur when shutting down
                memcpy((void*)trDgram, (const void*)dgram, sizeof(*dgram) - sizeof(dgram->xtc));
                if (transitionId != XtcData::TransitionId::SlowUpdate) {
                    // copy the temporary xtc created on phase 1 of the transition
                    // into the real location
                    XtcData::Xtc& trXtc = det->transitionXtc();
                    trDgram->xtc = trXtc; // Preserve header info, but allocate to check fit
                    auto payload = trDgram->xtc.alloc(trXtc.sizeofPayload(), bufEnd);
                    memcpy(payload, (const void*)trXtc.payload(), trXtc.sizeofPayload());
                }
                else {
                    det->slowupdate(trDgram->xtc, bufEnd);
                }
                // make sure the detector hasn't made the transition too big
                size_t size = sizeof(*trDgram) + trDgram->xtc.sizeofPayload();
                if (size > para.maxTrSize) {
                    logging::critical("%s: buffer size (%zd) too small for Dgram (%zd)",
                                      XtcData::TransitionId::name(transitionId), para.maxTrSize, size);
                    throw "Buffer too small";
                }
                if (trDgram->pulseId() != dgram->pulseId()) {
                    logging::critical("%s: pulseId (%014lx) doesn't match dgram's (%014lx)",
                                      XtcData::TransitionId::name(transitionId), trDgram->pulseId(), dgram->pulseId());
                }

                // Prepare the trigger primitive with whatever input is needed for the TEB to meke trigger decisions
                auto l3InpBuf = tebContributor.fetch(pebbleIndex);
                new(l3InpBuf) Pds::EbDgram(*dgram);
            }
        }

        outputQueue.push(batch);
    }
}

PGPDetector::PGPDetector(const Parameters& para, DrpBase& drp, Detector* det) :
    PgpReader(para, drp.pool, MAX_RET_CNT_C, para.batchSize), m_terminate(false),
    m_flushTmo(1.1 * drp.tebPrms().maxEntries * 14/13)
{
    m_nodeId = det->nodeId;
    if (drp.pool.setMaskBytes(para.laneMask, det->virtChan)) {
        logging::error("Failed to allocate lane/vc");
    }

    for (unsigned i=0; i<para.nworkers; i++) {
        m_workerInputQueues.emplace_back(SPSCQueue<Batch>(drp.pool.nbuffers()));
        m_workerOutputQueues.emplace_back(SPSCQueue<Batch>(drp.pool.nbuffers()));
    }


    for (unsigned i = 0; i < para.nworkers; i++) {
        m_workerThreads.emplace_back(workerFunc,
                                     std::ref(para),
                                     std::ref(drp),
                                     det,
                                     std::ref(m_workerInputQueues[i]),
                                     std::ref(m_workerOutputQueues[i]));
    }
}

PGPDetector::~PGPDetector()
{
    // Try to take things down gracefully when an exception takes us off the
    // normal path so that the most chance is given for prints to show up
    shutdown();
}

void PGPDetector::reader(std::shared_ptr<Pds::MetricExporter> exporter, Detector* det,
                         Pds::Eb::TebContributor& tebContributor)
{
    // setup monitoring
    uint64_t nevents = 0L;
    std::map<std::string, std::string> labels{{"instrument", m_para.instrument},
                                              {"partition", std::to_string(m_para.partition)},
                                              {"detname", m_para.detName},
                                              {"alias", m_para.alias}};
    exporter->add("drp_event_rate", labels, Pds::MetricType::Rate,
                  [&](){return nevents;});

    auto queueLength = [](std::vector<SPSCQueue<Batch> >& vec) {
        size_t sum = 0;
        for (auto& q: vec) {
            sum += q.guess_size();
        }
        return sum;
    };
    uint64_t nbuffers = m_para.nworkers * m_pool.nbuffers();
    exporter->constant("drp_worker_queue_depth", labels, nbuffers);

    exporter->add("drp_worker_input_queue", labels, Pds::MetricType::Gauge,
                  [&](){return queueLength(m_workerInputQueues);});

    exporter->add("drp_worker_output_queue", labels, Pds::MetricType::Gauge,
                  [&](){return queueLength(m_workerOutputQueues);});

    uint64_t nDmaRet = 0L;
    exporter->add("drp_num_dma_ret", labels, Pds::MetricType::Gauge,
                  [&](){return nDmaRet;});
    exporter->add("drp_pgp_byte_rate", labels, Pds::MetricType::Rate,
                  [&](){return dmaBytes();});
    exporter->add("drp_dma_size", labels, Pds::MetricType::Gauge,
                  [&](){return dmaSize();});
    exporter->add("drp_th_latency", labels, Pds::MetricType::Gauge,
                  [&](){return latency();});
    exporter->add("drp_num_dma_errors", labels, Pds::MetricType::Gauge,
                  [&](){return nDmaErrors();});
    exporter->add("drp_num_no_common_rog", labels, Pds::MetricType::Gauge,
                  [&](){return nNoComRoG();});
    exporter->add("drp_num_th_error", labels, Pds::MetricType::Gauge,
                  [&](){return nTmgHdrError();});
    exporter->add("drp_num_pgp_jump", labels, Pds::MetricType::Gauge,
                  [&](){return nPgpJumps();});
    exporter->add("drp_num_no_tr_dgram", labels, Pds::MetricType::Gauge,
                  [&](){return nNoTrDgrams();});

    int64_t worker = 0L;
    uint64_t batchId = 0L;
    resetEventCounter();

    enum TmoState { None, Started, Finished };
    TmoState tmoState(TmoState::None);
    const std::chrono::microseconds tmo(m_flushTmo);
    auto tInitial = Pds::fast_monotonic_clock::now(CLOCK_MONOTONIC);

    while (1) {
        if (m_terminate.load(std::memory_order_relaxed)) {
            break;
        }
        int32_t ret = read();
        nDmaRet = ret;
        if (ret == 0) {
            if (tmoState == TmoState::None) {
                tmoState = TmoState::Started;
                tInitial = Pds::fast_monotonic_clock::now(CLOCK_MONOTONIC);
            } else {
                if (Pds::fast_monotonic_clock::now(CLOCK_MONOTONIC) - tInitial > tmo) {
                    if (m_batch.size != 0) {
                        m_workerInputQueues[worker % m_para.nworkers].push(m_batch);
                        worker++;
                        m_batch.start += m_batch.size;
                        m_batch.size = 0;
                        batchId += m_para.batchSize;
                    } else {
                        if (tmoState != TmoState::Finished) {
                            m_workerInputQueues[worker % m_para.nworkers].push(m_batch);
                            worker++;
                            tmoState = TmoState::Finished;
                        }
                    }
                }
            }
        }
        for (int b=0; b < ret; b++) {
            tmoState = TmoState::None;
            const Pds::TimingHeader* timingHeader = handle(det, b);
            if (!timingHeader)  continue;

            nevents++;
            m_batch.size++;

            // send batch to worker if batch is full or if it's a transition
            XtcData::TransitionId::Value transitionId = timingHeader->service();
            if (((batchId ^ timingHeader->pulseId()) & ~(m_para.batchSize - 1)) ||
                ((transitionId != XtcData::TransitionId::L1Accept) &&
                 (transitionId != XtcData::TransitionId::SlowUpdate))) {
                m_workerInputQueues[worker % m_para.nworkers].push(m_batch);
                worker++;
                m_batch.start = timingHeader->evtCounter + 1;
                m_batch.size = 0;
                batchId = timingHeader->pulseId();
            }
        }
    }
    logging::info("PGPReader is exiting");
}

void PGPDetector::collector(Pds::Eb::TebContributor& tebContributor)
{
    int64_t worker = 0L;
    Batch batch;
    const unsigned bufferMask = m_pool.nDmaBuffers() - 1;
    while (true) {
        if (!m_workerOutputQueues[worker % m_para.nworkers].pop(batch)) {
            break;
        }
        for (unsigned i=0; i<batch.size; i++) {
            unsigned index = (batch.start + i) & bufferMask;
            PGPEvent* event = &m_pool.pgpEvents[index];
            if (event->mask == 0)
                continue;               // Skip broken event
            unsigned pebbleIndex = event->pebbleIndex;
            freeDma(event);
            tebContributor.process(pebbleIndex);
        }
        if (batch.size == 0) {
            tebContributor.timeout();
        }
        worker++;
    }
    logging::info("PGPCollector is exiting");
}

void PGPDetector::handleBrokenEvent(const PGPEvent& event)
{
    ++m_batch.size; // Broken events must be included in the batch since f/w advanced evtCounter
}

void PGPDetector::resetEventCounter()
{
    PgpReader::resetEventCounter();
    m_batch.start = 1;
    m_batch.size = 0;
}

void PGPDetector::shutdown()
{
    if (m_terminate.load(std::memory_order_relaxed))
        return;                         // Already shut down
    m_terminate.store(true, std::memory_order_release);
    logging::info("shutting down PGPReader");
    for (unsigned i = 0; i < m_para.nworkers; i++) {
        m_workerInputQueues[i].shutdown();
        if (m_workerThreads[i].joinable()) {
            m_workerThreads[i].join();
        }
    }
    logging::info("Worker threads finished");
    for (unsigned i = 0; i < m_para.nworkers; i++) {
        m_workerOutputQueues[i].shutdown();
    }

    // Flush the DMA buffers
    flush();
}
