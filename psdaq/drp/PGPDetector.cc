#include <iostream>
#include <atomic>
#include <limits.h>
#include <fcntl.h>
#include <sys/msg.h>
#include <sys/wait.h>
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
#include "psdaq/service/IpcUtils.hh"

#ifndef POSIX_TIME_AT_EPICS_EPOCH
#define POSIX_TIME_AT_EPICS_EPOCH 631152000u
#endif

using logging = psalg::SysLog;


using namespace Drp;
using namespace Pds::Ipc;

bool checkPulseIds(const Detector* det, PGPEvent* event)
{
    uint64_t pulseId = 0;
    for (int i=0; i<PGP_MAX_LANES; i++) {
        if (event->mask & (1 << i)) {
            uint32_t index = event->buffers[i].index;
            const Pds::TimingHeader* timingHeader = det->getTimingHeader(index);
            if (pulseId == 0) {
                pulseId = timingHeader->pulseId();
            }
            else {
                if (pulseId != timingHeader->pulseId()) {
                    logging::error("Wrong pulse id! expected %014lx but got %014lx instead",
                                   pulseId, timingHeader->pulseId());
                    return false;
                }
            }
            if (timingHeader->error()) {
                logging::error("Timing header error bit is set");
            }
        }
    }
    return true;
}

clockid_t test_coarse_clock() {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC_COARSE, &t) == 0) {
        return CLOCK_MONOTONIC_COARSE;
    } else {
        return CLOCK_MONOTONIC;
    }
}


void  drpSendReceive(int inpMqId, int resMqId, int inpShmId, int resShmId, void*& inpData, void*& resData,
                    XtcData::TransitionId::Value transitionId, unsigned threadNum)
{

    char msg[512];
    char recvmsg[520];

    if (transitionId == XtcData::TransitionId::Unconfigure) {
        logging::critical("[Thread %u] Unconfigure transition. Send stop message to Drp Python", threadNum);
        snprintf(msg, sizeof(msg), "%s", "s");
    } else {
        snprintf(msg, sizeof(msg), "%s", "g");
    }

    int rc = drpSend(inpMqId, msg, 1);
    if (rc) {
        logging::critical("[Thread %u] Error sending message %s to Drp python: %m",
                          threadNum, msg);
        abort();
    }

    rc = drpRecv(resMqId, recvmsg, sizeof(recvmsg), 10000);
    if (rc) {
        logging::critical("[Thread %u] Message from Drp python not received", threadNum);
        abort();
    }
}


void workerFunc(const Parameters& para, DrpBase& drp, Detector* det,
                SPSCQueue<Batch>& inputQueue, SPSCQueue<Batch>& outputQueue, bool pythonDrp,
                int inpMqId, int resMqId, int inpShmId, int resShmId, size_t shmemSize,
                unsigned threadNum, std::atomic<int>& threadCountPush, std::atomic<int>& threadCountWrite)
{
    Batch batch;
    MemPool& pool = drp.pool;
    const unsigned bufferMask = pool.nDmaBuffers() - 1;
    auto& tebContributor = drp.tebContributor();
    auto triggerPrimitive = drp.triggerPrimitive();
    void* inpData = nullptr;
    void* resData = nullptr;
    char msg[512];
    char recvmsg[520];
    bool transition;

    if (pythonDrp) {

        auto kwargs_it = para.kwargs.find("pythonScript");

        std::string pythonScript;
        if (kwargs_it != para.kwargs.end()) {
            pythonScript = kwargs_it->second;
        } else {
            logging::critical("[Thread %u] python drp script not specified" , threadNum);
            abort();
        }

        if (pythonScript.length() > 511) {
            logging::critical("[Thread %u] Path to python script too long (max 511 chars)" , threadNum);
            abort();
        }

        std::string keyBase = "p" + std::to_string(para.partition) + "_" + para.detName + "_" + std::to_string(para.detSegment);
        std::string key = "/shminp_" + keyBase + "_" + std::to_string(threadNum);

        int rc = attachDrpShMem(key, inpShmId, shmemSize, inpData, true);
        if (rc) {
            logging::critical("[Thread %u] error attaching to Drp shared memory buffer %s for key %u: %m",
                               threadNum, "Inputs", key);
            abort();
        }

        key = "/shmres_" + keyBase + "_" + std::to_string(threadNum);
        rc = attachDrpShMem(key, resShmId, shmemSize, resData, false);
        if (rc) {
            logging::critical("[Thread %u] error attaching to Drp shared memory buffer %s for key %u: %m",
                               threadNum, "Results", key);
            abort();
        }

        snprintf(msg, sizeof(msg), "%s",  pythonScript.c_str());

        rc = drpSend(inpMqId, msg, pythonScript.length());
        if (rc) {
            logging::critical("[Thread %u] Message %s from Drp python not sent", msg, threadNum);
            abort();
        }

        // Wait for python process to be up
        rc = drpRecv(resMqId, recvmsg, sizeof(recvmsg), 15000);
        if (rc) {
            logging::critical("[Thread %u] Message from Drp python not received", threadNum);
            abort();
        }

        logging::info("[Thread %u] Starting events", threadNum);
    }

    while (true) {

        if (!inputQueue.pop(batch)) {
            break;
        }

        transition=false;

        for (unsigned i=0; i<batch.size; i++) {

            unsigned index = (batch.start + i) & bufferMask;
            PGPEvent* event = &pool.pgpEvents[index];
            if (event->mask == 0)
                continue;               // Skip broken event

            // get transitionId from the first lane in the event
            int lane = __builtin_ffs(event->mask) - 1;
            uint32_t dmaIndex = event->buffers[lane].index;
            const Pds::TimingHeader* timingHeader = det->getTimingHeader(dmaIndex);

            unsigned pebbleIndex = event->pebbleIndex;
            XtcData::Src src = det->nodeId;
            XtcData::TransitionId::Value transitionId = timingHeader->service();

            // Event
            if (transitionId == XtcData::TransitionId::L1Accept) {
                // make new dgram in the pebble
                // It must be an EbDgram in order to be able to send it to the MEB
                Pds::EbDgram* dgram = new(pool.pebble[pebbleIndex]) Pds::EbDgram(*timingHeader, src, para.rogMask);

                const void* bufEnd = (char*)dgram + pool.bufferSize();
                det->event(*dgram, bufEnd, event);

                if ( pythonDrp) {
                    XtcData::Dgram* inpDg = dgram;
                    memcpy(inpData, (void*)inpDg, sizeof(*inpDg) + inpDg->xtc.sizeofPayload());
                    drpSendReceive(inpMqId, resMqId, inpShmId, resShmId, inpData, resData, transitionId, threadNum);
                    XtcData::Dgram* resDg = (XtcData::Dgram*)resData;
                    memcpy((void*)inpDg, resData, sizeof(*resDg) + resDg->xtc.sizeofPayload());
                }

                // Prepare the trigger primitive with whatever input is needed for the TEB to make trigger decisions
                auto l3InpBuf = tebContributor.fetch(pebbleIndex);
                Pds::EbDgram* l3InpDg = new(l3InpBuf) Pds::EbDgram(*dgram);

                if (triggerPrimitive) { // else this DRP doesn't provide input
                    const void* l3BufEnd = (char*)l3InpDg + sizeof(*l3InpDg) + triggerPrimitive->size();
                    triggerPrimitive->event(pool, pebbleIndex, dgram->xtc, l3InpDg->xtc, l3BufEnd);
                }
            // slow data
            } else if (transitionId == XtcData::TransitionId::SlowUpdate) {
                // make new dgram in the pebble
                // It must be an EbDgram in order to be able to send it to the MEB


                Pds::EbDgram* dgram = new(pool.pebble[pebbleIndex]) Pds::EbDgram(*timingHeader, src, para.rogMask);
                logging::debug("[Thread %u] PGPDetector saw %s @ %u.%09u (%014lx)",
                               threadNum,
                               XtcData::TransitionId::name(transitionId),
                               dgram->time.seconds(), dgram->time.nanoseconds(), dgram->pulseId());
                // Find the transition dgram in the pool and initialize its header
                Pds::EbDgram* trDgram = pool.transitionDgrams[pebbleIndex];
                const void*   bufEnd  = (char*)trDgram + para.maxTrSize;
                if (!trDgram)  continue; // Can occur when shutting down
                memcpy((void*)trDgram, (const void*)dgram, sizeof(*dgram) - sizeof(dgram->xtc));
                det->slowupdate(trDgram->xtc, bufEnd);

                if (pythonDrp) {
                    XtcData::Dgram* inpDg = trDgram;
                    memcpy(inpData, (void*)inpDg, sizeof(*inpDg) + inpDg->xtc.sizeofPayload());
                    drpSendReceive(inpMqId, resMqId, inpShmId, resShmId, inpData, resData, transitionId, threadNum);
                    XtcData::Dgram* resDg = (XtcData::Dgram*)(resData);
                    memcpy((void*)inpDg, resData, sizeof(*resDg) + resDg->xtc.sizeofPayload());
                }


                // Prepare the trigger primitive with whatever input is needed for the TEB to meke trigger decisions
                auto l3InpBuf = tebContributor.fetch(pebbleIndex);
                new(l3InpBuf) Pds::EbDgram(*dgram);
            // transitions
            } else {
                transition = true;
                // Pds::EbDgram* dgram = reinterpret_cast<Pds::EbDgram*>(pool.pebble[pebbleIndex]);
                Pds::EbDgram* trDgram = pool.transitionDgrams[pebbleIndex];
                if ( pythonDrp) {
                    XtcData::Dgram* inpDg = trDgram;
                    memcpy(inpData, (void*)inpDg, sizeof(*inpDg) + inpDg->xtc.sizeofPayload());
                    drpSendReceive(inpMqId, resMqId, inpShmId, resShmId, inpData, resData, transitionId, threadNum);
                    // TODO: Add comment explaining how this works
                    if (threadCountWrite.fetch_sub(1) == 1) {
                        XtcData::Dgram* resDg = (XtcData::Dgram*)(resData);
                        memcpy((void*)inpDg, resData, sizeof(*resDg) + resDg->xtc.sizeofPayload());
                    }
                }
            }
        }

        if (pythonDrp) {
            // TODO: Comment
            // All but the last worker to get here set the batch size to 0.
            // A batch size of 0 ensures collector runs, but doesn't do anything
            // other than advancing to the next worker.
            if (transition && threadCountPush.fetch_sub(1) != 1) {
                batch.size = 0;
            }
        }

        outputQueue.push(batch);
    }

    if (pythonDrp) {
        logging::info("[Thread %u] Detaching from Drp shared memory and message queues", threadNum);

        std::string keyBase = "p" + std::to_string(para.partition) + "_" + para.detName + "_" + std::to_string(para.detSegment);
        std::string key = "/shminp_" + keyBase + "_" + std::to_string(threadNum);
        int rc = detachDrpShMem(inpData, shmemSize);
        if (rc) {
            logging::critical("[Thread %u] error detaching from Drp shared memory buffer %s for key %u: %m",
                                threadNum, "Inputs", key);
            abort();
        }

        key = "/shmres_" + keyBase + "_" + std::to_string(threadNum);
        rc = detachDrpShMem(resData, shmemSize);
        if (rc) {
            logging::critical("[Thread %u] error detaching from Drp shared memory buffer %s for key %u: %m",
                                threadNum, "Results", key);
            abort();
        }

    }

}

PGPDetector::PGPDetector(const Parameters& para, DrpBase& drp, Detector* det,
                         bool pythonDrp, int* inpMqId, int* resMqId, int* inpShmId, int* resShmId,
                         size_t shmemSize) :
    PgpReader(para, drp.pool, MAX_RET_CNT_C, para.batchSize), m_terminate(false),
    m_flushTmo(1.1 * drp.tebPrms().maxEntries * 14/13),
    m_shmemSize(shmemSize),
    pythonDrp(pythonDrp)
{
    threadCountPush.store(0);
    threadCountWrite.store(0);
    m_nodeId = det->nodeId;
    int* m_inpMqId = inpMqId;
    int* m_resMqId = resMqId;
    int* m_inpShmId = inpShmId;
    int* m_resShmId = resShmId;

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
                                     std::ref(m_workerOutputQueues[i]),
                                     pythonDrp,
                                     m_inpMqId[i],
                                     m_resMqId[i],
                                     m_inpShmId[i],
                                     m_resShmId[i],
                                     m_shmemSize,
                                     i,
                                     std::ref(threadCountPush),
                                     std::ref(threadCountWrite));
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
    const unsigned bufferMask = m_pool.nDmaBuffers() - 1;

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
    exporter->add("drp_num_missing_rogs", labels, Pds::MetricType::Gauge,
                  [&](){return nMissingRoGs();});
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
                    // Time out partial DRP batches
                    if (m_batch.size != 0) {
                        m_workerInputQueues[worker % m_para.nworkers].push(m_batch);
                        worker++;
                        m_batch.start += m_batch.size;
                        m_batch.size = 0;
                        batchId += m_para.batchSize;
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

            bool stateTransition = (transitionId != XtcData::TransitionId::L1Accept) &&
                                   (transitionId != XtcData::TransitionId::SlowUpdate);

            // send batch to worker if batch is full or if it's a transition
            if (((batchId ^ timingHeader->pulseId()) & ~(m_para.batchSize - 1)) || stateTransition) {

                if ( stateTransition) {
                    if (m_batch.size > 1) {
                        m_batch.size--;
                        m_workerInputQueues[worker % m_para.nworkers].push(m_batch);
                        worker++;
                        m_batch.start += m_batch.size;
                        m_batch.size = 1;
                    }

                    unsigned index = m_batch.start & bufferMask;
                    PGPEvent* event = &m_pool.pgpEvents[index];
                    if (event->mask == 0)
                        continue;               // Skip broken event

                    unsigned pebbleIndex = event->pebbleIndex;
                    XtcData::Src src = det->nodeId;
                    Pds::EbDgram* dgram = new(m_pool.pebble[pebbleIndex]) Pds::EbDgram(*timingHeader,
                                            src, m_para.rogMask);

                    logging::debug("PGPDetector saw %s @ %u.%09u (%014lx)",
                                XtcData::TransitionId::name(transitionId),
                                dgram->time.seconds(), dgram->time.nanoseconds(),
                                dgram->pulseId());

                    // Initialize the transition dgram's header
                    Pds::EbDgram* trDgram = m_pool.transitionDgrams[pebbleIndex];
                    if (!trDgram)  continue; // Can occur when shutting down
                    memcpy((void*)trDgram, (const void*)dgram, sizeof(*dgram) - sizeof(dgram->xtc));

                    // copy the temporary xtc created on phase 1 of the transition
                    // into the real location
                    XtcData::Xtc& trXtc = det->transitionXtc();
                    trDgram->xtc = trXtc; // Preserve header info, but allocate to check fit
                    const void*   bufEnd  = (char*)trDgram + m_para.maxTrSize;
                    auto payload = trDgram->xtc.alloc(trXtc.sizeofPayload(), bufEnd);
                    memcpy(payload, (const void*)trXtc.payload(), trXtc.sizeofPayload());

                    // Prepare the trigger primitive with whatever input is needed for the TEB to meke trigger decisions
                    auto l3InpBuf = tebContributor.fetch(pebbleIndex);
                    new(l3InpBuf) Pds::EbDgram(*dgram);

                    // set thread counter and broadcast transition
                    threadCountWrite.store(m_para.nworkers);
                    threadCountPush.store(m_para.nworkers);

                    unsigned numWorkers = pythonDrp ? m_para.nworkers : 1;

                    for (unsigned w=0; w < numWorkers; w++) {
                        m_workerInputQueues[worker % m_para.nworkers].push(m_batch);
                        worker++;
                    }
                } else {
                    m_workerInputQueues[worker % m_para.nworkers].push(m_batch);
                    worker++;
                }

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
    bool rc = m_workerOutputQueues[worker % m_para.nworkers].pop(batch);
    while (rc) {
        for (unsigned i=0; i<batch.size; i++) {
            unsigned index = (batch.start + i) & bufferMask;
            PGPEvent* event = &m_pool.pgpEvents[index];
            if (event->mask == 0)
                continue;               // Skip broken event
            unsigned pebbleIndex = event->pebbleIndex;
            freeDma(event);
            tebContributor.process(pebbleIndex);
        }
        worker++;

        // Time out batches for the TEB
        while (!m_workerOutputQueues[worker % m_para.nworkers].try_pop(batch)) { // Poll
            if (tebContributor.timeout()) {                                      // After batch is timed out
                rc = m_workerOutputQueues[worker % m_para.nworkers].popW(batch); // pend
                break;
            }
        }
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
    for (unsigned i = 0; i < m_para.nworkers; i++) {
        m_workerOutputQueues[i].shutdown();
    }

    // Flush the DMA buffers
    flush();
}
