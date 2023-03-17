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



// int checkDrpPy(pid_t pid, bool wait = false)
// {
//   pid_t child_status = waitpid(pid, NULL, wait ? 0 : WNOHANG);
//   if (child_status != 0)
//   {
//     return -1;
//   }
//   return 0;
// }


void  sendReceiveDrp(int inpMqId, int resMqId, int inpShmId, int resShmId, void*& inpData, void*& resData,
                    clockid_t clockType, XtcData::TransitionId::Value transitionId, unsigned threadNum)
{
    Message_t msg;
    msg.mtype = 1;
    msg.mtext[0] = 'g';

    if (transitionId == XtcData::TransitionId::Unconfigure) {
        logging::critical("[Thread %u] Unconfigure transition. Send stop message to Drp Python", threadNum);
        msg.mtext[0] = 's';
    } else {
        msg.mtext[0] = 'g';
    }

    int rc = send(inpMqId, msg, 1, threadNum);
    if (rc) {
        logging::critical("[Thread %u] Message from Drp python not received", threadNum);
        // cleanupIpcPython(inpMqId, resMqId, inpShmId, resShmId, inpData, resData, threadNum); 
        abort();
    }

    rc = recv(resMqId, msg, 10000, clockType, threadNum);
    if (rc) {
        logging::critical("[Thread %u] Message from Drp python not received", threadNum);
        // cleanupIpcPython(inpMqId, resMqId, inpShmId, resShmId, inpData, resData, threadNum);
        abort();
    }
}


void workerFunc(const Parameters& para, DrpBase& drp, Detector* det,
                SPSCQueue<Batch>& inputQueue, SPSCQueue<Batch>& outputQueue,
                int inpMqId, int resMqId, int inpShmId, int resShmId,
                unsigned threadNum, std::atomic<int>& threadCount)
{
    Batch batch;
    MemPool& pool = drp.pool;
    const unsigned KEY_BASE = 40000;
    const unsigned dmaBufferMask = pool.nDmaBuffers() - 1;
    const unsigned pebbleBufferMask = pool.nbuffers() - 1;
    auto& tebContributor = drp.tebContributor();
    auto triggerPrimitive = drp.triggerPrimitive();
    auto& tebPrms = drp.tebPrms();
    bool pythonDrp = false;
    void* inpData = nullptr;
    void* resData = nullptr;
    Message_t msg;
    bool transition;

    clockid_t clockType = test_coarse_clock();

    auto kwargs_it = para.kwargs.find("drp");
    if (kwargs_it != para.kwargs.end() && kwargs_it->second == "python") {
        pythonDrp = true;
    }

    if (pythonDrp == true) {       

        kwargs_it = para.kwargs.find("pythonScript");

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

        unsigned keyBase  =  KEY_BASE + 1000 * threadNum + 100 * para.partition;

        int rc = attachDrpShMem(keyBase+2, "Inputs", inpShmId, inpData, threadNum);
        if (rc) {
            // cleanupDrpPython(inpMqId, resMqId, inpShmId, resShmId, inpData, resData, threadNum);
            logging::critical("[Thread %u] error setting up Drp shared memory buffers", threadNum);
            abort();
        }

        rc = attachDrpShMem(keyBase+3, "Results", resShmId, resData, threadNum);
        if (rc) {
            // cleanupDrpPython(inpMqId, resMqId, inpShmId, resShmId, inpData, resData, threadNum);
            logging::critical("[Thread %u] error setting up Drp shared memory buffers", threadNum);
            abort();
        }

        Message_t scriptmsg; 
        scriptmsg.mtype = 1;
        strncpy(scriptmsg.mtext, pythonScript.c_str(), pythonScript.length());

        rc = send(inpMqId, scriptmsg, pythonScript.length(), threadNum);
        if (rc) {
            logging::critical("[Thread %u] Message from Drp python not received", threadNum);
            // cleanupDrpPython(inpMqId, resMqId, inpShmId, resShmId, inpData, resData, threadNum);
            abort();
        }

        // Wait for python process to be up
        rc = recv(resMqId, msg, 15000, clockType, threadNum);
        if (rc) {
            logging::critical("[Thread %u] Message from Drp python not received", threadNum);
            // cleanupDrpPython(inpMqId, resMqId, inpShmId, resShmId, inpData, resData, threadNum);
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
            unsigned evtCounter = batch.start + i;
            PGPEvent* event = &pool.pgpEvents[evtCounter & dmaBufferMask];

            // get transitionId from the first lane in the event
            int lane = __builtin_ffs(event->mask) - 1;
            uint32_t dmaIndex = event->buffers[lane].index;
            const Pds::TimingHeader* timingHeader = det->getTimingHeader(dmaIndex);

            XtcData::TransitionId::Value transitionId = timingHeader->service();
            unsigned index = evtCounter & pebbleBufferMask;

            // Event
            if (transitionId == XtcData::TransitionId::L1Accept) {
                // make new dgram in the pebble
                // It must be an EbDgram in order to be able to send it to the MEB
                Pds::EbDgram* dgram = new(pool.pebble[index]) Pds::EbDgram(*timingHeader, XtcData::Src(det->nodeId), para.rogMask);
      
                const void* bufEnd = (char*)dgram + pool.bufferSize();
                det->event(*dgram, bufEnd, event);

                // make sure the detector hasn't made the event too big
                if (dgram->xtc.extent > pool.bufferSize()) {
                    logging::critical("[Thread %u] L1Accept: buffer size (%d) too small for requested extent (%d)", threadNum, pool.bufferSize(), dgram->xtc.extent);
                    throw "Buffer too small";
                }

                if ( pythonDrp == true) {
                    memcpy(inpData, (pool.pebble[index])+sizeof(Pds::PulseId), pool.pebble.bufferSize()-sizeof(Pds::PulseId));
                    sendReceiveDrp(inpMqId, resMqId, inpShmId, resShmId, inpData, resData, clockType, transitionId, threadNum);
                    memcpy((pool.pebble[index])+sizeof(Pds::PulseId), resData, pool.pebble.bufferSize()-sizeof(Pds::PulseId));
                }    

                // Prepare the trigger primitive with whatever input is needed for the TEB to meke trigger decisions
                auto l3InpBuf = tebContributor.fetch(index);
                Pds::EbDgram* l3InpDg = new(l3InpBuf) Pds::EbDgram(*dgram);
                
                if (triggerPrimitive) { // else this DRP doesn't provide input
                    const void* l3BufEnd = (char*)l3InpDg + sizeof(*l3InpDg) + triggerPrimitive->size();
                    triggerPrimitive->event(pool, index, dgram->xtc, l3InpDg->xtc, l3BufEnd);
                    size_t size = sizeof(*l3InpDg) + l3InpDg->xtc.sizeofPayload();
                    if (size > tebPrms.maxInputSize) {
                        logging::critical("[Thread %u] L3 Input Dgram of size %zd overflowed buffer of size %zd", threadNum, size, tebPrms.maxInputSize);
                        throw "Input Dgram overflowed buffer";
                    }
                }
            // slow data
            } else if (transitionId == XtcData::TransitionId::SlowUpdate) {
                Pds::EbDgram* dgram = new(pool.pebble[index]) Pds::EbDgram(*timingHeader, XtcData::Src(det->nodeId), para.rogMask);
                logging::debug("[Thread %u] PGPDetector saw %s @ %u.%09u (%014lx)",
                               threadNum,
                               XtcData::TransitionId::name(transitionId),
                               dgram->time.seconds(), dgram->time.nanoseconds(), dgram->pulseId());
                // Find the transition dgram in the pool and initialize its header
                Pds::EbDgram* trDgram = pool.transitionDgrams[index];
                const void*   bufEnd  = (char*)trDgram + para.maxTrSize;
                if (!trDgram)  continue; // Can occur when shutting down
                memcpy((void*)trDgram, (const void*)dgram, sizeof(*dgram) - sizeof(dgram->xtc));
                det->slowupdate(trDgram->xtc, bufEnd);
                // make sure the detector hasn't made the transition too big
                size_t size = sizeof(*trDgram) + trDgram->xtc.sizeofPayload();
                if (size > para.maxTrSize) {
                    logging::critical("[Thread %u] %s: buffer size (%zd) too small for Dgram (%zd)",
                                      threadNum, XtcData::TransitionId::name(transitionId), para.maxTrSize, size);
                    throw "Buffer too small";
                }

                if (trDgram->pulseId() != dgram->pulseId()) {
                    logging::critical("%s: pulseId (%014lx) doesn't match dgram's (%014lx)",
                                      XtcData::TransitionId::name(transitionId), trDgram->pulseId(), dgram->pulseId());
                }

                if ( pythonDrp == true) {
                    memcpy(inpData, ((char*)trDgram)+sizeof(Pds::PulseId), sizeof(XtcData::Dgram)+trDgram->xtc.extent);
                    sendReceiveDrp(inpMqId, resMqId, inpShmId, resShmId, inpData, resData, clockType, transitionId, threadNum);
                    XtcData::Dgram* resDgram = (XtcData::Dgram*)(resData);
                    memcpy(((char*)pool.transitionDgrams[index])+sizeof(Pds::PulseId), resData, sizeof(XtcData::Dgram) + resDgram->xtc.extent);
                }    

                // Prepare the trigger primitive with whatever input is needed for the TEB to meke trigger decisions
                auto l3InpBuf = tebContributor.fetch(index);
                new(l3InpBuf) Pds::EbDgram(*dgram);

            // transitions
            } else {
                transition = true;
                Pds::EbDgram* dgram = reinterpret_cast<Pds::EbDgram*>(pool.pebble[index]);
                Pds::EbDgram* trDgram = pool.transitionDgrams[index];
                logging::debug("[Thread %u] PGPDetector saw %s @ %u.%09u (%014lx)",
                               threadNum,
                               XtcData::TransitionId::name(transitionId),
                               dgram->time.seconds(), dgram->time.nanoseconds(), dgram->pulseId());
 
                if ( pythonDrp == true) {
                    memcpy(inpData, ((char*)trDgram)+sizeof(Pds::PulseId), sizeof(XtcData::Dgram)+trDgram->xtc.extent);
                    sendReceiveDrp(inpMqId, resMqId, inpShmId, resShmId, inpData, resData, clockType, transitionId, threadNum);
                    XtcData::Dgram* resDgram = (XtcData::Dgram*)(resData);
                    memcpy(((char*)pool.transitionDgrams[index])+sizeof(Pds::PulseId), resData, sizeof(XtcData::Dgram)+resDgram->xtc.extent);
                }
 
                auto l3InpBuf = tebContributor.fetch(index);
                if (threadNum == 0) {
                    new(l3InpBuf) Pds::EbDgram(*dgram);
                } 
                //else {
                //     Pds::EbDgram* dgram = reinterpret_cast<Pds::EbDgram*>(l3InpBuf);
                // }
            }
        }

        // only one thread sends a batch with content (size > 0) to the collector
        if (transition == true && threadCount.fetch_sub(1)!= 1) batch.size = 0;
        outputQueue.push(batch);
    }   
}

PGPDetector::PGPDetector(const Parameters& para, DrpBase& drp, Detector* det, int* inpMqId, int* resMqId, int* inpShmId, int* resShmId) :
    PgpReader(para, drp.pool, MAX_RET_CNT_C, para.batchSize), m_terminate(false)
{
    threadCount.store(0);
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
                                     m_inpMqId[i],
                                     m_resMqId[i],
                                     m_inpShmId[i],
                                     m_resShmId[i],
                                     i,
                                     std::ref(threadCount));
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
    const unsigned pebbleBufferMask = m_pool.nbuffers() - 1;
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
    m_batch.start = 0;
    m_batch.size = 0;
    resetEventCounter();

    while (1) {
         if (m_terminate.load(std::memory_order_relaxed)) {
            break;
        }
        int32_t ret = read();
        nDmaRet = ret;
        for (int b=0; b < ret; b++) {
            unsigned evtCounter;
            const Pds::TimingHeader* timingHeader = handle(det, b, evtCounter);
            if (!timingHeader)  continue;

            nevents++;
            m_batch.size++;

            // send batch to worker if batch is full or if it's a transition
            XtcData::TransitionId::Value transitionId = timingHeader->service();

            bool stateTransition = transitionId != XtcData::TransitionId::L1Accept && transitionId!=XtcData::TransitionId::SlowUpdate; 

            // send batch to worker if batch is full or if it's a transition
            if (((batchId ^ timingHeader->pulseId()) & ~(m_para.batchSize - 1)) || stateTransition == true ) {
                if (stateTransition == true) {
                    if (m_batch.size > 1) {
                        m_batch.size--;
                        m_workerInputQueues[worker % m_para.nworkers].push(m_batch);
                        worker++;
                        m_batch.start = (m_batch.start + m_batch.size);
                        m_batch.size = 1;
                    }
                    unsigned index = m_batch.start & pebbleBufferMask;
                    Pds::EbDgram* dgram = new(m_pool.pebble[index]) Pds::EbDgram(*timingHeader, XtcData::Src(det->nodeId), m_para.rogMask);

                    // Initialize the transition dgram's header
                    Pds::EbDgram* trDgram = m_pool.transitionDgrams[index];
                    if (!trDgram)  continue; // Can occur when shutting down
                    memcpy((void*)trDgram, (const void*)dgram, sizeof(*dgram) - sizeof(dgram->xtc));
                    
                    // copy the temporary xtc created on phase 1 of the transition
                    // into the real location
                    XtcData::Xtc& trXtc = det->transitionXtc();
                    trDgram->xtc = trXtc; // Preserve header info, but allocate to check fit
                    const void*   bufEnd  = (char*)trDgram + m_para.maxTrSize;
                    auto payload = trDgram->xtc.alloc(trXtc.sizeofPayload(), bufEnd);
                    memcpy(payload, (const void*)trXtc.payload(), trXtc.sizeofPayload());
                    
                    // make sure the detector hasn't made the transition too big
                    size_t size = sizeof(*trDgram) + trDgram->xtc.sizeofPayload();
                    if (size > m_para.maxTrSize) {
                        logging::critical("%s: buffer size (%zd) too small for Dgram (%zd)",
                                        XtcData::TransitionId::name(transitionId), m_para.maxTrSize, size);
                        throw "Buffer too small";
                    }
                    if (trDgram->pulseId() != dgram->pulseId()) {
                        logging::critical("%s: pulseId (%014lx) doesn't match dgram's (%014lx)",
                                        XtcData::TransitionId::name(transitionId), trDgram->pulseId(), dgram->pulseId());
                    }

                    // set thread counter and broadcast transition
                    threadCount.store(m_para.nworkers);
                    for (unsigned w=0; w < m_para.nworkers; w++) {
                        m_workerInputQueues[w].push(m_batch);
                    }
                } else {
                    m_workerInputQueues[worker % m_para.nworkers].push(m_batch);
                    worker++;
                }
                m_batch.start = (evtCounter + 1);
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
    const unsigned dmaBufferMask = m_pool.nDmaBuffers() - 1;
    const unsigned pebbleBufferMask = m_pool.nbuffers() - 1;
    while (true) {
        if (!m_workerOutputQueues[worker % m_para.nworkers].pop(batch)) {
            break;
        }
        for (unsigned i=0; i<batch.size; i++) {
            unsigned evtCounter = batch.start + i;
            PGPEvent* event = &m_pool.pgpEvents[evtCounter & dmaBufferMask];
            freeDma(event);
            tebContributor.process(evtCounter & pebbleBufferMask);
        }
        worker++;
    }
    logging::info("PGPCollector is exiting");
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
