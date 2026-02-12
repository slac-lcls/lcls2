#if !defined(_GNU_SOURCE)
#  define _GNU_SOURCE
#endif
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>

#include <iostream>
#include <atomic>
#include <limits.h>
#include <fcntl.h>
#include <sys/msg.h>
#include <sys/wait.h>
#include <sys/prctl.h>
#include "psdaq/aes-stream-drivers/DataDriver.h"
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
#include "TebReceiver.hh"
#include "CubeTebReceiver.hh"
#include "psdaq/service/IpcUtils.hh"

#ifndef POSIX_TIME_AT_EPICS_EPOCH
#define POSIX_TIME_AT_EPICS_EPOCH 631152000u
#endif

using json    = nlohmann::json;
using logging = psalg::SysLog;
using ns_t    = std::chrono::nanoseconds;


using namespace XtcData;
using namespace Drp;
using namespace Pds;
using namespace Pds::Ipc;


bool checkPulseIds(const Detector* det, PGPEvent* event)
{
    uint64_t pulseId = 0;
    for (int i=0; i<PGP_MAX_LANES; i++) {
        if (event->mask & (1 << i)) {
            uint32_t index = event->buffers[i].index;
            const TimingHeader* timingHeader = det->getTimingHeader(index);
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

static int drpSendReceive(int inpMqId, int resMqId, TransitionId::Value transitionId, unsigned threadNum)
{

    const char* msg;
    char recvmsg[520];

    if (transitionId == TransitionId::Unconfigure) {
        logging::debug("[Thread %u] Unconfigure transition. Send stop message to Drp Python", threadNum);
        msg = "s";
    } else {
        msg = "g";
    }

    int rc = drpSend(inpMqId, msg, 1);
    if (rc) {
        logging::error("[Thread %u] Error sending message %s to Drp python: %m", threadNum, msg);
        return rc;    // Return rather than abort so that teardown can happen
    }

    rc = drpRecv(resMqId, recvmsg, sizeof(recvmsg), 15000);
    if (rc) {
        logging::error("[Thread %u] Response message from Drp python not received: %m", threadNum);
        return rc;    // Return rather than abort so that teardown can happen
    }

    return rc;
}


void workerFunc(const Parameters& para, DrpBase& drp, Detector& det,
                SPSCQueue<Batch>& inputQueue, SPSCQueue<Batch>& outputQueue, bool pythonDrp,
                int inpMqId, int resMqId, int inpShmId, int resShmId, size_t shmemSize,
                unsigned threadNum, std::atomic<int>& threadCountPush, std::atomic<int>& threadCountWrite,
                int64_t& pythonTime)
{
    Batch batch;
    MemPool& pool = drp.pool;
    const unsigned bufferMask = pool.nDmaBuffers() - 1;
    void* inpData = nullptr;
    void* resData = nullptr;
    char recvmsg[520];
    bool transition;
    bool error = false;

    if (pythonDrp) {

        std::string keyBase = "p" + std::to_string(para.partition) + "_" + para.detName + "_" + std::to_string(para.detSegment);
        std::string key = "/shminp_" + keyBase + "_" + std::to_string(threadNum);

        int rc = attachDrpShMem(key, inpShmId, shmemSize, inpData, true);
        if (rc) {
            logging::error("[Thread %u] Error attaching to Drp shared memory buffer %s for key %s: %m",
                           threadNum, "Inputs", key.c_str());
            return;     // Return rather than abort so that teardown can happen
        }

        key = "/shmres_" + keyBase + "_" + std::to_string(threadNum);
        rc = attachDrpShMem(key, resShmId, shmemSize, resData, false);
        if (rc) {
            logging::error("[Thread %u] Error attaching to Drp shared memory buffer %s for key %s: %m",
                           threadNum, "Results", key.c_str());
            return;     // Return rather than abort so that teardown can happen
        }

        std::string message = (para.kwargs.find("pythonScript")->second + "," +
                               (drp.isSupervisor() ? "supervisor" : "") + "," +
                               drp.supervisorIpPort());

        rc = drpSend(inpMqId, message.c_str(), message.length());
        if (rc) {
            logging::error("[Thread %u] Message %s to Drp python not sent",
                           message.c_str(), threadNum);
            return;     // Return rather than abort so that teardown can happen
        }

        // Wait for python process to be up
        rc = drpRecv(resMqId, recvmsg, sizeof(recvmsg), 15000);
        if (rc) {
            logging::error("[Thread %u] 'Ready' message from Drp python not received", threadNum);
            return;     // Return rather than abort so that teardown can happen
        }

        logging::debug("[Thread %u] Starting events", threadNum);
    }

    pythonTime = 0ll;

    logging::info("Worker %u is starting with process ID %lu", threadNum, syscall(SYS_gettid));
    char nameBuf[16];
    snprintf(nameBuf, sizeof(nameBuf), "drp/Worker%d", threadNum);
    if (prctl(PR_SET_NAME, nameBuf, 0, 0, 0) == -1) {
        perror("prctl");
    }

    while (true) {

        if (!inputQueue.pop(batch)) [[unlikely]] {
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
            const TimingHeader* timingHeader = det.getTimingHeader(dmaIndex);

            unsigned pebbleIndex = event->pebbleIndex;

            Src src = det.nodeId;
            TransitionId::Value transitionId = timingHeader->service();

            // Event
            if (transitionId == TransitionId::L1Accept) {
                // make new dgram in the pebble
                // It must be an EbDgram in order to be able to send it to the MEB
                EbDgram* dgram = new(pool.pebble[pebbleIndex]) EbDgram(*timingHeader, src, para.rogMask);

                const void* bufEnd = (char*)dgram + pool.bufferSize();
                det.event(*dgram, bufEnd, event, ++batch.l1count);

                if ( pythonDrp) {
                    Dgram* inpDg = dgram;
                    memcpy(inpData, (void*)inpDg, sizeof(*inpDg) + inpDg->xtc.sizeofPayload());
                    auto t0{fast_monotonic_clock::now(CLOCK_MONOTONIC)};
                    auto rc = drpSendReceive(inpMqId, resMqId, transitionId, threadNum);
                    auto t1{fast_monotonic_clock::now(CLOCK_MONOTONIC)};
                    pythonTime = std::chrono::duration_cast<ns_t>(t1 - t0).count();
                    if (rc)  error = true;
                    Dgram* resDg = (Dgram*)resData;
                    memcpy((void*)inpDg, resData, sizeof(*resDg) + resDg->xtc.sizeofPayload());
                }

                // Prepare the trigger primitive with whatever input is needed for the TEB to make trigger decisions
                auto l3InpBuf = drp.tebContributor().fetch(pebbleIndex);
                EbDgram* l3InpDg = new(l3InpBuf) EbDgram(*dgram);

                auto trgPrimitive = drp.triggerPrimitive();
                if (trgPrimitive) { // else this DRP doesn't provide input
                    const void* l3BufEnd = (char*)l3InpDg + sizeof(*l3InpDg) + trgPrimitive->size();
                    trgPrimitive->event(pool, pebbleIndex, dgram->xtc, l3InpDg->xtc, l3BufEnd);
                }
            // slow data
            } else if (transitionId == TransitionId::SlowUpdate) {
                // make new dgram in the pebble
                // It must be an EbDgram in order to be able to send it to the MEB


                EbDgram* dgram = new(pool.pebble[pebbleIndex]) EbDgram(*timingHeader, src, para.rogMask);
                logging::debug("PGPDetector saw %s @ %u.%09u (%014lx) [Thread %u]",
                               TransitionId::name(transitionId),
                               dgram->time.seconds(), dgram->time.nanoseconds(), dgram->pulseId(), threadNum);
                // Find the transition dgram in the pool and initialize its header
                EbDgram* trDgram = pool.transitionDgrams[pebbleIndex];
                const void*   bufEnd  = (char*)trDgram + para.maxTrSize;
                if (!trDgram)  continue; // Can occur when shutting down
                memcpy((void*)trDgram, (const void*)dgram, sizeof(*dgram) - sizeof(dgram->xtc));
                det.slowupdate(trDgram->xtc, bufEnd);

                if (pythonDrp) {
                    Dgram* inpDg = trDgram;
                    memcpy(inpData, (void*)inpDg, sizeof(*inpDg) + inpDg->xtc.sizeofPayload());
                    auto rc = drpSendReceive(inpMqId, resMqId, transitionId, threadNum);
                    if (rc)  error = true;
                    Dgram* resDg = (Dgram*)(resData);
                    memcpy((void*)inpDg, resData, sizeof(*resDg) + resDg->xtc.sizeofPayload());
                }

                // Prepare the trigger primitive with whatever input is needed for the TEB to meke trigger decisions
                auto l3InpBuf = drp.tebContributor().fetch(pebbleIndex);
                new(l3InpBuf) EbDgram(*dgram);
            // transitions
            } else {
                transition = true;
                EbDgram* trDgram = pool.transitionDgrams[pebbleIndex];

                //  Allow trigger primitives to parse Configure/Names data
                if (transitionId == TransitionId::Configure) {
                    auto trgPrimitive = drp.triggerPrimitive();
                    if (trgPrimitive) { // else this DRP doesn't provide input
                        trgPrimitive->configure(trDgram->xtc, (char*)trDgram + para.maxTrSize);
                    }
                }

                if (pythonDrp) {
                    Dgram* inpDg = trDgram;
                    memcpy(inpData, (void*)inpDg, sizeof(*inpDg) + inpDg->xtc.sizeofPayload());
                    auto rc = drpSendReceive(inpMqId, resMqId, transitionId, threadNum);
                    if (rc)  error = true;
                    // TODO: Add comment explaining how this works
                    if (!error && threadCountWrite.fetch_sub(1) == 1) {
                        Dgram* resDg = (Dgram*)(resData);
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

        if (!error)  outputQueue.push(batch);
    }

    if (pythonDrp) {
        logging::debug("[Thread %u] Detaching from Drp shared memory and message queues", threadNum);

        std::string keyBase = "p" + std::to_string(para.partition) + "_" + para.detName + "_" + std::to_string(para.detSegment);
        std::string key = "/shminp_" + keyBase + "_" + std::to_string(threadNum);
        int rc = detachDrpShMem(inpData, shmemSize);
        if (rc) {
            logging::error("[Thread %u] Error detaching from Drp shared memory buffer %s for key %s: %m",
                              threadNum, "Inputs", key.c_str());
            // Even on error, go on to try to detach from /shmres
        }

        key = "/shmres_" + keyBase + "_" + std::to_string(threadNum);
        rc = detachDrpShMem(resData, shmemSize);
        if (rc) {
            logging::error("[Thread %u] Error detaching from Drp shared memory buffer %s for key %s: %m",
                              threadNum, "Results", key.c_str());
            // Even on error, continue so that teardown can complete
        }
    }

    logging::info("Worker %u is exiting", threadNum);
}

// ---

Pgp::Pgp(const Parameters& para, MemPool& pool, Detector& det, PGPDrp& drp) :
   PgpReader(para, pool, MAX_RET_CNT_C, para.batchSize),
   m_drp(drp)
{
    if (pool.setMaskBytes(para.laneMask, det.virtChan)) {
        logging::critical("Failed to allocate lane/vc: '%m' "
                          "- does another process have %s open?", para.device.c_str());
        abort();
    }
}

void Pgp::handleBrokenEvent(const PGPEvent& event)
{
    m_drp.handleBrokenEvent(event);
}

void Pgp::resetEventCounter()
{
    PgpReader::resetEventCounter();
    m_drp.resetEventCounter();
}

// ---

PGPDrp::PGPDrp(Parameters& para, MemPool& pool, Detector& det, ZmqContext& context,
               int* inpMqId, int* resMqId, int* inpShmId, int* resShmId, size_t shmemSize) :
    DrpBase    (para, pool, det, context),
    m_para     (para),
    m_det      (det),
    m_pgp      (para, pool, det, *this),
    m_terminate(false),
    m_inpMqId  (inpMqId),
    m_resMqId  (resMqId),
    m_inpShmId (inpShmId),
    m_resShmId (resShmId),
    m_flushTmo (1.1 * tebPrms().maxEntries * 14/13),
    m_shmemSize(shmemSize),
    m_pyAppTime(0),
    m_pythonDrp(false)
{
    // Set the TebReceiver we will use in the base class
    if (para.nCubeWorkers==0)
        setTebReceiver(std::make_unique<TebReceiver>(m_para, *this));
    else
        setTebReceiver(std::make_unique<CubeTebReceiver>(m_para, *this));
}

std::string PGPDrp::configure(const json& msg)
{
    std::string errorMsg = DrpBase::configure(msg);
    if (!errorMsg.empty()) {
        return errorMsg;
    }

    auto kwargs_it = m_para.kwargs.find("drp");
    bool pythonDrp = kwargs_it != m_para.kwargs.end() && kwargs_it->second == "python";

    // Python-DRP is disabled during calibrations
    const std::string& config_alias = msg["body"]["config_alias"];
    m_pythonDrp = config_alias != "CALIB" ? pythonDrp : false;
    m_terminate.store(false, std::memory_order_release);

    m_pythonDrp = pythonDrp;

    threadCountPush.store(0);
    threadCountWrite.store(0);

    if (m_pythonDrp) {
        auto kwargs_it = m_para.kwargs.find("pythonScript");

        std::string pythonScript;
        if (kwargs_it != m_para.kwargs.end()) {
            pythonScript = kwargs_it->second;
        } else {
            errorMsg = "Python drp script not specified";
            return errorMsg;
        }

        if (pythonScript.length() > 511) {
            errorMsg = "Path to python script too long (max 511 chars)";
            return errorMsg;
        }
    }

    for (unsigned i=0; i<m_para.nworkers; i++) {
        m_workerInputQueues.emplace_back(SPSCQueue<Batch>(pool.nbuffers()));
        m_workerOutputQueues.emplace_back(SPSCQueue<Batch>(pool.nbuffers()));
    }

    for (unsigned i = 0; i < m_para.nworkers; i++) {
        m_workerThreads.emplace_back(workerFunc,
                                     std::ref(m_para),
                                     std::ref(*this),
                                     std::ref(m_det),
                                     std::ref(m_workerInputQueues[i]),
                                     std::ref(m_workerOutputQueues[i]),
                                     m_pythonDrp,
                                     m_inpMqId[i],
                                     m_resMqId[i],
                                     m_inpShmId[i],
                                     m_resShmId[i],
                                     m_shmemSize,
                                     i,
                                     std::ref(threadCountPush),
                                     std::ref(threadCountWrite),
                                     std::ref(m_pyAppTime));
    }

    m_pgpThread = std::thread{&PGPDrp::reader, std::ref(*this)};
    m_collectorThread = std::thread(&PGPDrp::collector, std::ref(*this));

    return std::string();
}

unsigned PGPDrp::unconfigure()
{
    DrpBase::unconfigure(); // TebContributor must be shut down before the worker

    m_terminate.store(true, std::memory_order_release);

    if (m_workerThreads.size())
        logging::info("Shutting down workers");
    for (unsigned i = 0; i < m_workerThreads.size(); i++) {
        m_workerInputQueues[i].shutdown();
        if (m_workerThreads[i].joinable()) {
            m_workerThreads[i].join();
        }
    }
    if (m_workerThreads.size())
        logging::info("Worker threads finished");

    for (unsigned i = 0; i < m_workerOutputQueues.size(); i++) {
        m_workerOutputQueues[i].shutdown();
    }

    if (m_pgpThread.joinable()) {
        m_pgpThread.join();
        logging::info("PGPReader thread finished");
    }
    if (m_collectorThread.joinable()) {
        m_collectorThread.join();
        logging::info("Collector thread finished");
    }

    m_workerInputQueues.clear();
    m_workerThreads.clear();
    m_workerOutputQueues.clear();

    return 0;
}

int PGPDrp::_setupMetrics(const std::shared_ptr<MetricExporter> exporter)
{
    std::map<std::string, std::string> labels{{"instrument", m_para.instrument},
                                              {"partition", std::to_string(m_para.partition)},
                                              {"detname", m_para.detName},
                                              {"alias", m_para.alias}};
    m_nevents = 0L;
    exporter->add("drp_event_rate", labels, MetricType::Rate,
                  [&](){return m_nevents;});

    auto queueLength = [](std::vector<SPSCQueue<Batch> >& vec) {
        size_t sum = 0;
        for (auto& q: vec) {
            sum += q.guess_size();
        }
        return sum;
    };
    uint64_t nbuffers = m_para.nworkers * pool.nbuffers();
    exporter->constant("drp_worker_queue_depth", labels, nbuffers);

    exporter->add("drp_worker_input_queue", labels, MetricType::Gauge,
                  [&](){return queueLength(m_workerInputQueues);});

    exporter->add("drp_worker_output_queue", labels, MetricType::Gauge,
                  [&](){return queueLength(m_workerOutputQueues);});

    m_nDmaRet = 0L;
    exporter->add("drp_num_dma_ret", labels, MetricType::Gauge,
                  [&](){return m_nDmaRet;});
    exporter->add("drp_pgp_byte_rate", labels, MetricType::Rate,
                  [&](){return m_pgp.dmaBytes();});
    exporter->add("drp_dma_size", labels, MetricType::Gauge,
                  [&](){return m_pgp.dmaSize();});
    exporter->add("drp_th_latency", labels, MetricType::Gauge,
                  [&](){return m_pgp.latency();});
    exporter->add("drp_num_dma_errors", labels, MetricType::Gauge,
                  [&](){return m_pgp.nDmaErrors();});
    exporter->add("drp_num_no_common_rog", labels, MetricType::Gauge,
                  [&](){return m_pgp.nNoComRoG();});
    exporter->add("drp_num_missing_rogs", labels, MetricType::Gauge,
                  [&](){return m_pgp.nMissingRoGs();});
    exporter->add("drp_num_th_error", labels, MetricType::Gauge,
                  [&](){return m_pgp.nTmgHdrError();});
    exporter->add("drp_num_pgp_jump", labels, MetricType::Gauge,
                  [&](){return m_pgp.nPgpJumps();});
    exporter->add("drp_num_no_tr_dgram", labels, MetricType::Gauge,
                  [&](){return m_pgp.nNoTrDgrams();});

    exporter->constant("drp_num_pgp_bufs", labels, pool.dmaCount());
    exporter->add("drp_num_pgp_in_user", labels, MetricType::Gauge,
                  [&](){return m_pgp.nPgpInUser();});
    exporter->add("drp_num_pgp_in_hw", labels, MetricType::Gauge,
                  [&](){return m_pgp.nPgpInHw();});
    exporter->add("drp_num_pgp_in_prehw", labels, MetricType::Gauge,
                  [&](){return m_pgp.nPgpInPreHw();});
    exporter->add("drp_num_pgp_in_rx", labels, MetricType::Gauge,
                  [&](){return m_pgp.nPgpInRx();});

    if (m_pythonDrp) {
        exporter->add("drp_py_app_time", labels, MetricType::Gauge,
                      [&](){return m_pyAppTime;});
    }

    return 0;
}

void PGPDrp::reader()
{
    const unsigned bufferMask = pool.nDmaBuffers() - 1;
    int64_t worker = 0L;
    uint64_t batchId = 0L;
    uint64_t pendingL1 = 0L;

    enum TmoState { None, Started, Finished };
    TmoState tmoState(TmoState::None);
    const std::chrono::microseconds tmo(m_flushTmo);
    auto tInitial = fast_monotonic_clock::now(CLOCK_MONOTONIC);

    logging::info("PGP reader is starting with process ID %lu", syscall(SYS_gettid));
    if (prctl(PR_SET_NAME, "drp/PGPreader", 0, 0, 0) == -1) {
        perror("prctl");
    }

    // If triggers had been left running, they will have been stopped during Allocate
    // Flush anything that accumulated
    m_pgp.flush();

    // Reset counters to avoid 'jumping' errors on reconfigures
    pool.resetCounters();
    m_pgp.resetEventCounter();

    // Set up monitoring
    auto exporter = std::make_shared<MetricExporter>();
    if (exposer()) {
        exposer()->RegisterCollectable(exporter);

        if (_setupMetrics(exporter))  return;
    }

    while (true) {
         if (m_terminate.load(std::memory_order_relaxed)) [[unlikely]] {
            break;
        }
        int32_t ret = m_pgp.read();
        m_nDmaRet = ret;

        // Time out and flush DRP and TEB batches when there no DMAed data came in
        if (ret == 0) {
            if (tmoState == TmoState::None) {
                tmoState = TmoState::Started;
                tInitial = fast_monotonic_clock::now(CLOCK_MONOTONIC);
            } else {
                if (fast_monotonic_clock::now(CLOCK_MONOTONIC) - tInitial > tmo) {
                    // Time out partial DRP batches
                    if (m_batch.size != 0) {
                        m_workerInputQueues[worker % m_para.nworkers].push(m_batch);
                        worker++;
                        m_batch.start += m_batch.size;
                        m_batch.size = 0;
                        batchId += m_para.batchSize;
                        // now that dispatched the batch to worker add the pending L1 count for the next batch
                        m_batch.l1count += pendingL1;
                        pendingL1 = 0;
                    }
                }
            }
        }

        for (int b=0; b < ret; b++) {
            tmoState = TmoState::None;
            const TimingHeader* timingHeader = m_pgp.handle(&m_det, b);
            if (!timingHeader)  continue;

            m_nevents++;
            m_batch.size++;

            // send batch to worker if batch is full or if it's a transition
            TransitionId::Value transitionId = timingHeader->service();

            bool stateTransition = (transitionId != TransitionId::L1Accept) &&
                                   (transitionId != TransitionId::SlowUpdate);

            // keep track of the number of L1Accepts seen in the batch
            if (transitionId == TransitionId::L1Accept) {
                pendingL1++;
            }

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
                    PGPEvent* event = &pool.pgpEvents[index];
                    if (event->mask == 0)
                        continue;               // Skip broken event

                    unsigned pebbleIndex = event->pebbleIndex;
                    Src src = nodeId();
                    EbDgram* dgram = new(pool.pebble[pebbleIndex]) EbDgram(*timingHeader,
                                                                           src, m_para.rogMask);

                    logging::debug("PGPDrp saw %s @ %u.%09u (%014lx)",
                                   TransitionId::name(transitionId),
                                   dgram->time.seconds(), dgram->time.nanoseconds(),
                                   dgram->pulseId());

                    // Initialize the transition dgram's header
                    EbDgram* trDgram = pool.transitionDgrams[pebbleIndex];
                    if (!trDgram)  continue; // Can occur when shutting down
                    memcpy((void*)trDgram, (const void*)dgram, sizeof(*dgram) - sizeof(dgram->xtc));

                    // copy the temporary xtc created on phase 1 of the transition
                    // into the real location
                    Xtc& trXtc = m_det.transitionXtc();
                    trDgram->xtc = trXtc; // Preserve header info, but allocate to check fit
                    const void*   bufEnd  = (char*)trDgram + m_para.maxTrSize;
                    auto payload = trDgram->xtc.alloc(trXtc.sizeofPayload(), bufEnd);
                    memcpy(payload, (const void*)trXtc.payload(), trXtc.sizeofPayload());

                    // Prepare the trigger primitive with whatever input is needed for the TEB to meke trigger decisions
                    auto l3InpBuf = tebContributor().fetch(pebbleIndex);
                    new(l3InpBuf) EbDgram(*dgram);

                    // set thread counter and broadcast transition
                    threadCountWrite.store(m_para.nworkers);
                    threadCountPush.store(m_para.nworkers);

                    unsigned numWorkers = m_pythonDrp ? m_para.nworkers : 1;

                    for (unsigned w=0; w < numWorkers; w++) {
                        m_workerInputQueues[worker % m_para.nworkers].push(m_batch);
                        worker++;
                    }
                } else {                // L1Accept or SlowUpdate
                    m_workerInputQueues[worker % m_para.nworkers].push(m_batch);
                    worker++;
                }

                m_batch.start = timingHeader->evtCounter + 1;
                m_batch.size = 0;
                batchId = timingHeader->pulseId();
                // now that dispatched the batch to worker add the pending L1 count for the next batch
                m_batch.l1count += pendingL1;
                pendingL1 = 0;
            }
        }
    }

    // Flush the PGP Reader buffers
    m_pgp.flush();

    if (exposer()) {
        exporter.reset();
    }

    logging::info("PGPReader is exiting");
}

void PGPDrp::collector()
{
    logging::info("Collector is starting with process ID %lu", syscall(SYS_gettid));
    if (prctl(PR_SET_NAME, "drp/Collector", 0, 0, 0) == -1) {
        perror("prctl");
    }

    int64_t worker = 0L;
    Batch batch;
    const unsigned bufferMask = pool.nDmaBuffers() - 1;
    bool rc = m_workerOutputQueues[worker % m_para.nworkers].pop(batch);
    while (rc) {
         if (m_terminate.load(std::memory_order_relaxed)) [[unlikely]] {
            break;
        }
        for (unsigned i=0; i<batch.size; i++) {
            unsigned index = (batch.start + i) & bufferMask;
            PGPEvent* event = &pool.pgpEvents[index];
            if (event->mask == 0)
                continue;               // Skip broken event
            unsigned pebbleIndex = event->pebbleIndex;
            m_pgp.freeDma(event);
            tebContributor().process(pebbleIndex);
        }
        worker++;

        // Time out batches for the TEB
        while (!m_workerOutputQueues[worker % m_para.nworkers].try_pop(batch)) { // Poll
            if (tebContributor().timeout()) {                                    // After batch is timed out
                rc = m_workerOutputQueues[worker % m_para.nworkers].popW(batch); // pend
                break;
            }
        }
    }
    logging::info("Collector is exiting");
}

void PGPDrp::handleBrokenEvent(const PGPEvent& event)
{
    ++m_batch.size; // Broken events must be included in the batch since f/w advanced evtCounter
}

void PGPDrp::resetEventCounter()
{
    m_batch.start = 1;
    m_batch.size = 0;
    m_batch.l1count = 0;
}
