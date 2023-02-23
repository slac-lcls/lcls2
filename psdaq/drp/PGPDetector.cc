#include <iostream>
#include <atomic>
#include <limits.h>
#include <fcntl.h>
#include <sys/ipc.h>
#include <sys/shm.h>
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

#ifndef POSIX_TIME_AT_EPICS_EPOCH
#define POSIX_TIME_AT_EPICS_EPOCH 631152000u
#endif

using logging = psalg::SysLog;
using ms_t = std::chrono::milliseconds;

using namespace Drp;

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

struct Message_t
{
    long mtype;                       // message type
    char mtext[512];                  // message text
};

clockid_t test_coarse_clock() {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC_COARSE, &t) == 0) {
        return CLOCK_MONOTONIC_COARSE;
    } else {
        return CLOCK_MONOTONIC;
    }
}

int startDrpPython(pid_t& pyPid, unsigned threadNum,
                   std::string pythonScript, unsigned keyBase, long shmemSize)
{


  // Fork
  pyPid = fork();

  if (pyPid == pid_t(0))
  {
    logging::info("[Thread %u] Running 'python -u %s'", threadNum, pythonScript.c_str());

    // Executing external code 
    int rc = execlp("python",
                    "python",
                    "-u",
                    pythonScript.c_str(),
                    std::to_string(keyBase).c_str(),
                    std::to_string(shmemSize).c_str(),
                    nullptr);
    // Execlp returns only on error                    
    logging::error("[Thread %u] Error on exec 'python -u %s %s ': %m",
                   threadNum, pythonScript, std::to_string(threadNum));
    return rc;
  } else {
    return 0;
  }
}

int checkDrpPy(pid_t pid, bool wait = false)
{
  pid_t child_status = waitpid(pid, NULL, wait ? 0 : WNOHANG);
  if (child_status != 0)
  {
    return -1;
  }
  return 0;
}


// void drainDrpPipe(int pipe, unsigned threadNum)
// {

//   char read_buffer[1024];
//   int nbytes;

//   int flags = fcntl(pipe, F_GETFL);
//   flags |= O_NONBLOCK;
//   int ret_val = fcntl(pipe, F_SETFL, flags);
//   if (ret_val < 0)
//   {
//     logging::error("[Thread %u] fcntl failed: %m", threadNum);
//     return;
//   }

//   while (true)
//   {
//     nbytes = read(pipe, read_buffer, sizeof(read_buffer));
//     if (nbytes > 0)
//     {
//       read_buffer[nbytes] = '\0';
//       logging::info("[Python, Thread %u] <<< %s >>>", threadNum, read_buffer);
//     }
//     else if (nbytes == 0)
//     {
//       return;
//     }
//     else if (nbytes < 0 && (errno == EWOULDBLOCK || errno == EAGAIN))
//     {
//       return;
//     }
//     else
//     {
//       logging::error("[Thread %u] Error reading from pipe: %s: %m", threadNum, read_buffer);
//       logging::error("[Thread %u] Exit error %d: %m", threadNum, nbytes);
//       return;
//     }
//   }
// }

void cleanupDrpPython(int inpMqId, int resMqId, int inpShmId, int resShmId, void*& inpData, void*& resData,
                      unsigned threadNum)
{
    if (inpMqId) {
        msgctl(inpMqId, IPC_RMID, NULL);
        inpMqId  = 0;
    }
    if (resMqId)   {
        msgctl(resMqId, IPC_RMID, NULL); 
        resMqId  = 0;
    }
    if (inpData) {
        shmdt(inpData);
        inpData = nullptr;
    }
    if (inpShmId) {
        shmctl(inpShmId, IPC_RMID, NULL);
        inpShmId = 0;
    }
    if (resData) {
        shmdt (resData);
        resData  = nullptr;
    }
    if (resShmId) {
        shmctl(resShmId, IPC_RMID, NULL); 
        resShmId = 0;
    }
}

int setupDrpShMem(key_t key, size_t size, const char* name, int& shmId, void*& data, unsigned threadNum)
{
    shmId = shmget(key, size, IPC_CREAT | 0666); // IPC_EXCL
    if (shmId == -1)
    {
        logging::error("[Thread %u] Error in creating Drp %s shared memory for key %u: %m",
                        threadNum, name, key);
        return -1;
    }

    data = shmat(shmId, nullptr, 0);
    if (data == (void *)-1)
    {
        logging::error("[Thread %u] Error attaching Drp %s shared memory for key %u: %m",
                       threadNum, name, key);
        return -1;
    }

    logging::info("[Thread %u] %s shared memory created for key %u", threadNum, name, key);

    return 0;
}

int setupDrpMsgQueue(key_t key, const char* name, int& mqId, unsigned threadNum)
{
    mqId = msgget(key, IPC_CREAT | 0666);
    if (mqId == -1)
    {
        logging::error("[Thread %u] Error in creating Drp %s message queue with key %u: %m",
                       threadNum, name, key);
        return -1;
    }

    logging::info("[Thread %u] %s message queues created", threadNum, name);

    return 0;
}

int send(int mqId, const Message_t& msg, size_t size, unsigned threadNum)
{
    int rc = msgsnd(mqId, (void *)&msg, size, 0);
    if (rc == -1)
    {
        logging::error("[Thread %u] Error sending message '%c' to Drp python: %m",
                       threadNum, msg.mtext[0]);
        return -1;
    }
    return 0;
}

int recv(int mqId, Message_t& msg, unsigned msTmo, clockid_t clockType, unsigned threadNum)
{

    struct timespec t;
    [[maybe_unused]] auto result = clock_gettime(clockType, &t);
    assert(result == 0);
    auto tp = std::chrono::seconds(t.tv_sec) + std::chrono::nanoseconds(t.tv_nsec);

    while (true)
    {
        auto rc = msgrcv(mqId, &msg, sizeof(msg.mtext), 0, IPC_NOWAIT);
        if (rc != -1)  break;

        if (errno != ENOMSG)
        {
            logging::error("[Thread %u] Error receiving message from Drp python: %m", threadNum);
            return -1;
        }

        result = clock_gettime(clockType, &t);
        assert(result == 0);
        auto now = std::chrono::seconds(t.tv_sec) + std::chrono::nanoseconds(t.tv_nsec);
        
        auto dt  = std::chrono::duration_cast<ms_t>(now - tp).count();
        
        if (dt > msTmo)
        {
            logging::error("[Thread %u] Message receiving timed out", threadNum);
            return -1;
        }
    }
    return 0;
}

void  sendReceiveDrp(int inpMqId, int resMqId, int inpShmId, int resShmId, void*& inpData, void*& resData,
                    pid_t pyPid, clockid_t clockType, unsigned threadNum)
{
    Message_t msg;
    msg.mtype = 1;
    msg.mtext[0] = 'g';

    logging::info("[Thread %u] sending data to drp python", threadNum);

    int rc = send(inpMqId, msg, 1, threadNum);
    if (rc) {
        logging::critical("[Thread %u] Message from Drp python not received", threadNum);
        rc = checkDrpPy(pyPid);
        if (rc) {
            logging::critical("[Thread %u] drp python not running", threadNum);
            cleanupDrpPython(inpMqId, resMqId, inpShmId, resShmId, inpData, resData,
            threadNum);
        }
        abort();
    }

    rc = recv(resMqId, msg, 10000, clockType, threadNum);
    if (rc) {
        logging::critical("[Thread %u] Message from Drp python not received", threadNum);
        rc = checkDrpPy(pyPid);
        if (rc) {
            logging::critical("[Thread %u] drp python not running", threadNum);
            cleanupDrpPython(inpMqId, resMqId, inpShmId, resShmId, inpData, resData, threadNum);
        }
        abort();
    }
}

void workerFunc(const Parameters& para, DrpBase& drp, Detector* det,
                SPSCQueue<Batch>& inputQueue, SPSCQueue<Batch>& outputQueue,
                unsigned threadNum, std::atomic<int>& threadCount)
{
    Batch batch;
    MemPool& pool = drp.pool;
    const unsigned KEY_BASE = 40000;
    const unsigned bufferMask = pool.nbuffers() - 1;
    auto& tebContributor = drp.tebContributor();
    auto triggerPrimitive = drp.triggerPrimitive();
    auto& tebPrms = drp.tebPrms();
    bool pythonDrp = false;
    pid_t pyPid = 0;
    int inpMqId = 0;
    int resMqId = 0;
    int inpShmId = 0;
    int resShmId = 0;
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

        pythonDrp = true;
        unsigned keyBase  =  KEY_BASE + 1000 * threadNum + 100 * para.partition;

        // Creating message queues
        logging::info("[Thread %u] Creating Drp message queues" , threadNum);

        int rc = setupDrpMsgQueue(keyBase+0, "Inputs", inpMqId, threadNum);
        if (rc) {
            cleanupDrpPython(inpMqId, resMqId, inpShmId, resShmId, inpData, resData, threadNum);
            logging::critical("[Thread %u] error setting up Drp message queues", threadNum);
            abort();
        }
        rc = setupDrpMsgQueue(keyBase+1, "Results", resMqId, threadNum);
        if (rc) {
            cleanupDrpPython(inpMqId, resMqId, inpShmId, resShmId,  inpData, resData, threadNum);
            logging::critical("[Thread %u] error setting up Drp message queues", threadNum);
            abort();
        }

        // Creating shared memory
        logging::info("[Thread %u] Creating Drp shared memory blocks", threadNum);

        // Calculate the size of the Inputs data block
        // WARNING: the -8 at the end is the size of the pulseID
        // If the size of the pulseid changes, this number will
        // need to be changed
        size_t shmemSize = pool.pebble.bufferSize() - 8;
        
        // Round up to an integral number of pages
        long pageSize = sysconf(_SC_PAGESIZE);
        shmemSize = (shmemSize + pageSize - 1) & ~(pageSize - 1);

        rc = setupDrpShMem(keyBase+2, shmemSize, "Inputs", inpShmId, inpData, threadNum);
        if (rc) {
            cleanupDrpPython(inpMqId, resMqId, inpShmId, resShmId, inpData, resData, threadNum);
            logging::critical("[Thread %u] error setting up Drp shared memory buffers", threadNum);
            abort();
        }

        rc = setupDrpShMem(keyBase+3, shmemSize, "Results", resShmId, resData, threadNum);
        if (rc) {
            cleanupDrpPython(inpMqId, resMqId, inpShmId, resShmId, inpData, resData, threadNum);
            logging::critical("[Thread %u] error setting up Drp shared memory buffers", threadNum);
            abort();
        }

        // Fork
        logging::info("[Thread %u] Starting Drp Python process", threadNum);
        rc = startDrpPython(pyPid, threadNum, pythonScript, keyBase, shmemSize);
        if (rc || (pyPid == pid_t(0))) {
            logging::critical("[Thread %u] error starting Drp python process", threadNum);
            abort();
        }

        logging::info("[Thread %u] Waiting for python process to start", threadNum);

        // Wait for python process to be up
        rc = recv(resMqId, msg, 10000, clockType, threadNum);
        if (rc) {
            logging::critical("[Thread %u] Message from Drp python not received", threadNum);
            rc = checkDrpPy(pyPid);
            if (rc) {
                logging::critical("[Thread %u] drp python not running", threadNum);
                cleanupDrpPython(inpMqId, resMqId, inpShmId, resShmId, inpData, resData, threadNum);
            }
            abort();
        }
        logging::info("[Thread %u] Python process started", threadNum);
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
            checkPulseIds(det, event);

            // get transitionId from the first lane in the event
            int lane = __builtin_ffs(event->mask) - 1;
            uint32_t dmaIndex = event->buffers[lane].index;
            const Pds::TimingHeader* timingHeader = det->getTimingHeader(dmaIndex);

            XtcData::TransitionId::Value transitionId = timingHeader->service();

            logging::info("[Thread %u] DEBUG: Event coming through: %d", threadNum, transitionId);

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
                    memcpy(inpData, (pool.pebble[index])+8, pool.pebble.bufferSize());
                    sendReceiveDrp(inpMqId, resMqId, inpShmId, resShmId, inpData, resData, pyPid, clockType, threadNum);
                }    

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
                               dgram->time.seconds(), dgram->time.nanoseconds(), timingHeader->pulseId());
                // Initialize the transition dgram's header
                Pds::EbDgram* trDgram = event->transitionDgram;
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
                if (event->transitionDgram->pulseId() != dgram->pulseId()) {
                    logging::critical("[Thread %u] %s: pulseId (%014lx) doesn't match dgram's (%014lx)",
                                      threadNum, XtcData::TransitionId::name(transitionId), event->transitionDgram->pulseId(), dgram->pulseId());
                }

                if ( pythonDrp == true) {
                    memcpy(inpData, (pool.pebble[index])+8, pool.pebble.bufferSize());
                    sendReceiveDrp(inpMqId, resMqId, inpShmId, resShmId, inpData, resData, pyPid, clockType, threadNum);
                }    

                auto l3InpBuf = tebContributor.fetch(index);
                new(l3InpBuf) Pds::EbDgram(*dgram);

            // transitions
            } else {
                transition = true;
                Pds::EbDgram* dgram = reinterpret_cast<Pds::EbDgram*>(pool.pebble[index]);
                logging::debug("[Thread %u] PGPDetector saw %s @ %u.%09u (%014lx)",
                               threadNum,
                               XtcData::TransitionId::name(transitionId),
                               dgram->time.seconds(), dgram->time.nanoseconds(), timingHeader->pulseId());
 
                if ( pythonDrp == true) {
                    memcpy(inpData, (pool.pebble[index])+8, pool.pebble.bufferSize());
                    sendReceiveDrp(inpMqId, resMqId, inpShmId, resShmId, inpData, resData, pyPid, clockType, threadNum);
                }    
 
                auto l3InpBuf = tebContributor.fetch(index);
                if (threadNum == 0) {
                    new(l3InpBuf) Pds::EbDgram(*dgram);
                } else {
                    Pds::EbDgram* dgram = reinterpret_cast<Pds::EbDgram*>(l3InpBuf);
                }
            }
        }

        // only one thread sends a batch with content (size > 0) to the collector
        if (transition == true && threadCount.fetch_sub(1)!= 1) batch.size = 0;
        outputQueue.push(batch);
    }
        

    // if (pythonDrp == true) {

    //     logging::info("[Thread %u] Asking Drp python to stop", threadNum);
    //     msg.mtype = 1;
    //     msg.mtext[0] = 's';
    //     int rc = send(inpMqId, msg, 1, threadNum);
    //     if (rc) {
    //         logging::critical("[Thread %u] Message from Drp python not received", threadNum);
    //         rc = checkDrpPy(pyPid);
    //         drainDrpPipe(pipefd_stdout[0], threadNum);
    //         drainDrpPipe(pipefd_stderr[0], threadNum);
    //         if (rc) {
    //             logging::critical("[Thread %u] drp python not running", threadNum);
    //             cleanupDrpPython(inpMqId, resMqId, inpShmId, resShmId, inpData, resData, pipefd_stdout, pipefd_stdout, threadNum);
    //         }
    //         abort();
    //     }

    //     logging::info("[Thread %u] Waiting for Drp python confirmation", threadNum);
    //     rc = recv(resMqId, msg, 5000, clockType, threadNum);
    //     if (rc) {
    //         logging::critical("[Thread %u] Message from Drp python not received", threadNum);
    //         rc = checkDrpPy(pyPid);
    //         drainDrpPipe(pipefd_stdout[0], threadNum);
    //         drainDrpPipe(pipefd_stderr[0], threadNum);
    //         if (rc) {
    //             logging::critical("[Thread %u] drp python not running", threadNum);
    //             cleanupDrpPython(inpMqId, resMqId, inpShmId, resShmId, inpData, resData, pipefd_stdout, pipefd_stdout, threadNum);
    //         }
    //         abort();
    //     }

    //     rc = checkDrpPy(pyPid);
    //     if (!rc) {
    //         logging::critical("[Thread %u] drp python failed to stop", threadNum);
    //         abort();
    //     } else {
    //         logging::info("[Thread %u] Drp python stopped", threadNum);
    //     }
    // }       
    }


PGPDetector::PGPDetector(const Parameters& para, DrpBase& drp, Detector* det) :
    m_para(para), m_pool(drp.pool), m_terminate(false)
{
    threadCount.store(0);
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
                                     std::ref(m_workerOutputQueues[i]),
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
    uint64_t bytes = 0L;
    std::map<std::string, std::string> labels{{"instrument", m_para.instrument},
                                              {"partition", std::to_string(m_para.partition)},
                                              {"detname", m_para.detName},
                                              {"alias", m_para.alias}};
    exporter->add("drp_event_rate", labels, Pds::MetricType::Rate,
                  [&](){return nevents;});

    exporter->add("drp_pgp_byte_rate", labels, Pds::MetricType::Rate,
                  [&](){return bytes;});

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
    uint64_t dmaSize = 0L;
    exporter->add("drp_dma_size", labels, Pds::MetricType::Gauge,
                  [&](){return dmaSize;});

    int64_t latency = 0L;
    exporter->add("drp_th_latency", labels, Pds::MetricType::Gauge,
                  [&](){return latency;});

    int64_t worker = 0L;
    uint64_t batchId = 0L;
    const unsigned bufferMask = m_pool.nbuffers() - 1;
    XtcData::TransitionId::Value lastTid = XtcData::TransitionId::Reset;
    double lastTime = 0;
    uint32_t lastData[6];
    memset(lastData,0,sizeof(lastData));
    resetEventCounter();


    while (1) {
        if (m_terminate.load(std::memory_order_relaxed)) {
            break;
        }
        int32_t ret = dmaReadBulkIndex(m_pool.fd(), MAX_RET_CNT_C, dmaRet, dmaIndex, dmaFlags, dmaErrors, dest);
        nDmaRet = ret;
        for (int b=0; b < ret; b++) {
            uint32_t size = dmaRet[b];
            uint32_t index = dmaIndex[b];
            uint32_t lane = (dest[b] >> 8) & 7;
            dmaSize = size;
            bytes += size;
            if (size > m_pool.dmaSize()) {
                logging::critical("DMA overflowed buffer: %u vs %u", size, m_pool.dmaSize());
                throw "DMA overflowed buffer";
            }

            uint32_t flag = dmaFlags[b];
            uint32_t err  = dmaErrors[b];
            if (err) {
                logging::error("DMA with error 0x%x  flag 0x%x",err,flag);
                //  How do I return this buffer?
                dmaRetIndex(m_pool.fd(), index);
                nevents++;
                continue;
            }

            const Pds::TimingHeader* timingHeader = det->getTimingHeader(index);
            uint32_t evtCounter = timingHeader->evtCounter & 0xffffff;
            uint32_t current = evtCounter & bufferMask;
            PGPEvent* event = &m_pool.pgpEvents[current];

            DmaBuffer* buffer = &event->buffers[lane];
            buffer->size = size;
            buffer->index = index;
            event->mask |= (1 << lane);
            m_pool.allocate(1);

            const uint32_t* data = reinterpret_cast<const uint32_t*>(timingHeader);
            if (m_para.verbose < 2)
                logging::debug("PGPReader  lane %u  size %u  hdr %016lx.%016lx.%08x  flag 0x%x  err 0x%x",
                               lane, size,
                               reinterpret_cast<const uint64_t*>(data)[0],
                               reinterpret_cast<const uint64_t*>(data)[1],
                               reinterpret_cast<const uint32_t*>(data)[4],
                               flag, err);

            if (event->mask == m_para.laneMask) {
                bool stateTransition = false;
                XtcData::TransitionId::Value transitionId = timingHeader->service();
                if (transitionId != XtcData::TransitionId::L1Accept) {
                    if (transitionId!=XtcData::TransitionId::SlowUpdate) {
                        stateTransition = true;
                        logging::info("PGPReader  saw %s @ %u.%09u (%014lx)",
                                      XtcData::TransitionId::name(transitionId),
                                      timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                                      timingHeader->pulseId());
                    } else {
                        logging::debug("PGPReader  saw %s @ %u.%09u (%014lx)",
                                       XtcData::TransitionId::name(transitionId),
                                       timingHeader->time.seconds(), timingHeader->time.nanoseconds(),
                                       timingHeader->pulseId());
                    }
                    if (transitionId == XtcData::TransitionId::BeginRun) {
                        resetEventCounter();
                    }
                }
                if (evtCounter != ((m_lastComplete + 1) & 0xffffff)) {
                    logging::critical("%sPGPReader: Jump in complete l1Count %u -> %u | difference %d, tid %s%s",
                                      RED_ON, m_lastComplete, evtCounter, evtCounter - m_lastComplete, XtcData::TransitionId::name(transitionId), RED_OFF);
                    logging::critical("data: %08x %08x %08x %08x %08x %08x service: 0x%x",
                                      data[0], data[1], data[2], data[3], data[4], data[5], timingHeader->service());

                    logging::critical("lastTid %s", XtcData::TransitionId::name(lastTid));
                    logging::critical("lastData: %08x %08x %08x %08x %08x %08x",
                                      lastData[0], lastData[1], lastData[2], lastData[3], lastData[4], lastData[5]);

                    //  Do we still need to throw an exception?
                    //  Sometimes we have genuine frame errors
                    if (transitionId != XtcData::TransitionId::L1Accept ||
                        (timingHeader->time.asDouble()-lastTime)>1. ||
                        (timingHeader->time.asDouble()-lastTime)<0.) {
                        logging::critical("Jump in event counter");
                        abort();
                    }

                    for (unsigned e=m_lastComplete+1; e!=evtCounter; e++) {
                        PGPEvent* brokenEvent = &m_pool.pgpEvents[e & bufferMask];
                        logging::error("broken event:  %08x", brokenEvent->mask);
                        brokenEvent->mask = 0;
                        m_batch.size++; // Broken events are included in the batch
                    }
                }
                m_lastComplete = evtCounter;
                lastTime = timingHeader->time.asDouble();
                lastTid = transitionId;
                memcpy(lastData, data, 24);

                nevents++;
                m_batch.size++;

                // Allocate a transition datagram from the pool.  Since a
                // SPSCQueue is used (not an SPMC queue), this can be done here,
                // but not in the workers or there will be concurrency issues.
                if (transitionId != XtcData::TransitionId::L1Accept) {
                    event->transitionDgram = m_pool.allocateTr();
                    if (!event->transitionDgram)  break; // Can happen during shutdown
                }

                auto now = std::chrono::system_clock::now();
                auto dgt = std::chrono::seconds{timingHeader->time.seconds() + POSIX_TIME_AT_EPICS_EPOCH}
                         + std::chrono::nanoseconds{timingHeader->time.nanoseconds()};
                std::chrono::system_clock::time_point tp{std::chrono::duration_cast<std::chrono::system_clock::duration>(dgt)};
                latency = std::chrono::duration_cast<ms_t>(now - tp).count();

                // send batch to worker if batch is full or if it's a transition
                if (((batchId ^ timingHeader->pulseId()) & ~(m_para.batchSize - 1)) || stateTransition == true ) {
                    if (stateTransition == true) {
                        if (m_batch.size > 1) {
                            m_batch.size--;
                            m_workerInputQueues[worker % m_para.nworkers].push(m_batch);
                            worker++;
                            m_batch.start = (m_batch.start + m_batch.size) & 0xffffff;
                            m_batch.size = 1;
                        }
                        index = m_batch.start & bufferMask;
                        Pds::EbDgram* dgram = new(m_pool.pebble[index]) Pds::EbDgram(*timingHeader, XtcData::Src(det->nodeId), m_para.rogMask);
                        PGPEvent* event = &m_pool.pgpEvents[index];

                        // Initialize the transition dgram's header
                        Pds::EbDgram* trDgram = event->transitionDgram;
                        const void*   bufEnd  = (char*)trDgram + m_para.maxTrSize;
                        memcpy((void*)trDgram, (const void*)dgram, sizeof(*dgram) - sizeof(dgram->xtc));
                        
                        // copy the temporary xtc created on phase 1 of the transition
                        // into the real location
                        XtcData::Xtc& trXtc = det->transitionXtc();
                        trDgram->xtc = trXtc; // Preserve header info, but allocate to check fit
                        auto payload = trDgram->xtc.alloc(trXtc.sizeofPayload(), bufEnd);
                        memcpy(payload, (const void*)trXtc.payload(), trXtc.sizeofPayload());
                        
                        // make sure the detector hasn't made the transition too big
                        size_t size = sizeof(*trDgram) + trDgram->xtc.sizeofPayload();
                        if (size > m_para.maxTrSize) {
                            logging::critical("%s: buffer size (%zd) too small for Dgram (%zd)",
                                              XtcData::TransitionId::name(transitionId), m_para.maxTrSize, size);
                            throw "Buffer too small";
                        }
                        if (event->transitionDgram->pulseId() != dgram->pulseId()) {
                            logging::critical("%s: pulseId (%014lx) doesn't match dgram's (%014lx)",
                                              XtcData::TransitionId::name(transitionId), event->transitionDgram->pulseId(), dgram->pulseId());
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
                    m_batch.start = evtCounter + 1;
                    m_batch.size = 0;
                    batchId = timingHeader->pulseId();
                }
            }
        }
    }
    logging::info("PGPReader is exiting");
}

void PGPDetector::collector(Pds::Eb::TebContributor& tebContributor)
{
    int64_t worker = 0L;
    Batch batch;
    const unsigned bufferMask = m_pool.nbuffers() - 1;
    while (true) {
        if (!m_workerOutputQueues[worker % m_para.nworkers].pop(batch)) {
            break;
        }
        for (unsigned i=0; i<batch.size; i++) {
            unsigned index = (batch.start + i) & bufferMask;
            PGPEvent* event = &m_pool.pgpEvents[index];
            if (event->mask == 0)
                continue;               // Skip broken event
            tebContributor.process(index);
        }
        worker++;
    }
}

void PGPDetector::resetEventCounter()
{
    m_lastComplete = 0;
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

    //  Flush the DMA buffers
    int32_t ret = dmaReadBulkIndex(m_pool.fd(), MAX_RET_CNT_C, dmaRet, dmaIndex, NULL, NULL, dest);
    dmaRetIndexes(m_pool.fd(), ret, dmaIndex);
}
