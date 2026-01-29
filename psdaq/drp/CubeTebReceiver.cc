#include "CubeTebReceiver.hh"

#include "psalg/utils/SysLog.hh"
#include "psdaq/service/kwargs.hh"
#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/Smd.hh"
#include "psdaq/eb/CubeResultDgram.hh"

#include <sys/prctl.h>

using namespace XtcData;
using namespace Drp;
using namespace Pds;
using namespace Pds::Eb;
using logging = psalg::SysLog;
using us_t = std::chrono::microseconds;

namespace Drp {
    class BinDef : public VarDef {
    public:
        enum { bin, entries };
        BinDef()
        {
            Alg bin("bin", 1, 0, 0);
            NameVec.push_back({"bin", Name::UINT32, bin});
            NameVec.push_back({"entries", Name::UINT32, entries});
        }
    };
};

void workerFunc(const Parameters& para, Detector& det, MemPool& pool,
                std::vector<CubeResultDgram>& results,
                std::vector<char*>& bin_data,
                std::vector<unsigned>& bin_entries,
                Pds::Semaphore sem,
                SPSCQueue<unsigned>& inputQueue, SPSCQueue<unsigned>& outputQueue,
                unsigned threadNum)
{
    unsigned index;

    logging::info("CubeWorker %u is starting with process ID %lu", threadNum, syscall(SYS_gettid));
    char nameBuf[16];
    snprintf(nameBuf, sizeof(nameBuf), "drp/CubeWrk%d", threadNum);
    if (prctl(PR_SET_NAME, nameBuf, 0, 0, 0) == -1) {
        perror("prctl");
    }

    while (true) {

        if (!inputQueue.pop(index)) [[unlikely]] {
                break;
            }

        const CubeResultDgram& result = results[index];
        TransitionId::Value transitionId = result.service();
        auto dgram = transitionId == TransitionId::L1Accept ? (EbDgram*)pool.pebble[index]
            : pool.transitionDgrams[index];

        logging::warning("CubeWorker %u %s index (%u)",
                         threadNum, TransitionId::name(transitionId), index);
        // Event
        if (transitionId == TransitionId::L1Accept && result.persist()) {
            unsigned ibin = result.bin();
            sem.take();  // not bin-specific
            // add the event into the bin
            // the bin storage has to contain shape, thus the shapesdata;
            det.cube(*dgram, ibin, bin_data[ibin]);
            bin_entries[result.bin()]++;
            sem.give();
        }

        outputQueue.push(index);
    }

    logging::info("CubeWorker %u is exiting", threadNum);
}

#define BUFFER_SIZE m_pool.bufferSize()*10  // uint8_t -> double

CubeTebReceiver::CubeTebReceiver(const Parameters& para, DrpBase& drp) :
    TebReceiver  (para, drp),
    m_det        (drp.detector()),
    m_result     (m_pool.nbuffers(), CubeResultDgram(EbDgram(PulseId(0),Dgram()),0)),
    m_current    (-1),
    m_last       (-1),
    m_bin_data   (MAX_CUBE_BINS+1),
    m_bin_entries(para.nCubeWorkers, std::vector<unsigned>(MAX_CUBE_BINS,0)),
    m_buffer     (new char[BUFFER_SIZE]),
    m_sem        (para.nCubeWorkers,Pds::Semaphore(Pds::Semaphore::FULL)),
    m_terminate  (false)
{
     for(unsigned i=0; i<MAX_CUBE_BINS+1; i++)
        m_bin_data[i] = new char[BUFFER_SIZE];
}

//
//  Handles result
//    Queues binning summation to workers, if indicated
//    Post to monitoring
//
void CubeTebReceiver::complete(unsigned index, const ResultDgram& res)
{
    // This function is called by the base class's process() method to complete
    // processing and dispose of the event.  It presumes that the caller has
    // already vetted index and result
    const Pds::Eb::CubeResultDgram& result = reinterpret_cast<const Pds::Eb::CubeResultDgram&>(res);
    logging::debug("CubeTebReceiver::complete index (%u) result data(%x) aux(%x) persist(%u) monitor(%x) bin(%u) worker(%u) record(%u)", 
                   index, result.data(), result.auxd(), result.persist(), result.monitor(), result.bin(), result.worker(), result.record());

    TransitionId::Value transitionId = result.service();
    auto dgram = transitionId == TransitionId::L1Accept ? (EbDgram*)m_pool.pebble[index]
        : m_pool.transitionDgrams[index];

    m_evtSize = sizeof(Dgram) + dgram->xtc.sizeofPayload();

    // Measure latency before sending dgram for monitoring
    if (dgram->pulseId() - m_latPid > 1300000/14) { // 10 Hz
        m_latency = Pds::Eb::latency<us_t>(dgram->time);
        m_latPid = dgram->pulseId();
    }

    if (m_mon.enabled()) {
        // L1Accept
        if (result.isEvent()) {
            if (result.monitor()) {
                m_mon.post(dgram, result.monBufNo());
            }
        }
        // Other Transition
        else {
            m_mon.post(dgram);
        }
    }

    _queueDgram(index, result); // copies the result
}

void CubeTebReceiver::_queueDgram(unsigned index, const Pds::Eb::CubeResultDgram& result)
{
    TransitionId::Value transitionId = result.service();
    // auto dgram = transitionId == TransitionId::L1Accept ? (EbDgram*)m_pool.pebble[index]
    //                                                     : m_pool.transitionDgrams[index];
    if (transitionId != TransitionId::L1Accept) {
        if (transitionId == TransitionId::Configure) {
            m_collectorThread = std::thread(&CubeTebReceiver::process, std::ref(*this));
            for (unsigned i=0; i<m_para.nCubeWorkers; i++) {
                m_workerInputQueues .emplace_back(SPSCQueue<unsigned>(m_pool.nbuffers()));
                m_workerOutputQueues.emplace_back(SPSCQueue<unsigned>(m_pool.nbuffers()));
            }
            for (unsigned i=0; i<m_para.nCubeWorkers; i++) {
                m_workerThreads.emplace_back(workerFunc,
                                             std::ref(m_para),
                                             std::ref(m_det),
                                             std::ref(m_pool),
                                             std::ref(m_result),
                                             std::ref(m_bin_data),
                                             std::ref(m_bin_entries[i]),
                                             std::ref(m_sem[i]),
                                             std::ref(m_workerInputQueues[i]),
                                             std::ref(m_workerOutputQueues[i]),
                                             i);
            }
        }
        else if (transitionId == TransitionId::Unconfigure) {
            for(unsigned i=0; i<m_para.nCubeWorkers; i++) {
                m_workerInputQueues[i].shutdown();
                if (m_workerThreads[i].joinable())
                    m_workerThreads[i].join();
            }
            for(unsigned i=0; i<m_para.nCubeWorkers; i++) {
                m_workerOutputQueues[i].shutdown();
            }

            m_terminate.store(true, std::memory_order_release);
            if (m_collectorThread.joinable()) {
                m_collectorThread.join();
            }
            return;
        }
    }
    
    m_result[index] = result; // must copy (full xtc not copied!)
    m_last = index;
    
    unsigned worker = result.worker();
    logging::debug("CubeTebReceiver::queueDgram pushed index (%u) to worker (%u)",
                   index, worker);
    m_workerInputQueues[worker].push(index);
}

XtcData::Dgram* CubeTebReceiver::_binDgram(const CubeResultDgram& result)
{
    unsigned ibin = result.bin();
    //  Maybe build xtc with binId, entries, array here, too.
    XtcData::Dgram* dg = new(m_buffer) Dgram(result);
    NamesId namesId(m_det.nodeId, CubeNamesIndex);
    CreateData data(dg->xtc, m_buffer+BUFFER_SIZE, m_namesLookup, namesId);
    data.set_value<uint32_t>(CubeDef::bin, ibin);
    data.set_value<uint32_t>(CubeDef::entries, m_bin_entries[ibin]);
    Array<double_t>& arrayB = *reinterpret_cast<Array<double_t>*>(m_bin_data[ibin]);
    Array<double_t> arrayT = data.allocate<double_t>(CubeDef::array,arrayB.shape());
    memcpy(arrayT.data(), arrayB.data(), sizeof(double_t)*arrayB.num_elems());

    logging::debug("binDgram bin (%u) worker (%u) pulseId (%lu/0x%016lx) timeStamp (%lu/0x%016lx) extent (%u)",
                   result.bin(), result.worker(), 
                   result.pulseId(), result.pulseId(), 
                   result.time.value(), result.time.value(), 
                   result.xtc.extent);
    return dg;
}

void CubeTebReceiver::process()
{
    logging::warning("CubeTebReceiver::process is starting with process ID %lu", syscall(SYS_gettid));
    char nameBuf[16];
    snprintf(nameBuf, sizeof(nameBuf), "drp/CubeTebProc");
    if (prctl(PR_SET_NAME, nameBuf, 0, 0, 0) == -1) {
        perror("prctl");
    }

    while (true) {
        if (m_terminate.load(std::memory_order_relaxed)) [[unlikely]] {
            break;
        }

        if (m_current == m_last)
            continue;

        m_current = (m_current + 1) % (m_pool.nbuffers() - 1);

        unsigned index = m_current;
        //  Need to make sure the worker is done with this buffer
        const CubeResultDgram& result = m_result[index];

        logging::warning("CubeTebReceiver::process index (%u) result(%x) bin (%u) worker (%u)",
                         index, result.data(), result.bin(), result.worker());

        unsigned windex;
        bool rc = m_workerOutputQueues[result.worker()].pop(windex);
        if (rc) {

            logging::warning("CubeTebReceiver::process index %u  windex %u",
                             index,windex);

            TransitionId::Value transitionId = result.service();
            auto dgram = transitionId == TransitionId::L1Accept ? 
                (EbDgram*)m_pool.pebble[index] : m_pool.transitionDgrams[index];

            if (writing()) {                    // Won't ever be true for Configure
                if (result.persist() || result.prescale()) {
                    if (result.record()) {
                        //  Write the intermediate accumulated bin
                        m_sem[result.worker()].take();
                        _writeDgram(_binDgram(result));
                        m_sem[result.worker()].give();
                    }
                    else
                        //  Do I need to record an empty dgram for the offline event builder?
                        _writeDgram(const_cast<CubeResultDgram*>(&result));
                }
                else if (transitionId != TransitionId::L1Accept) {
                    if (transitionId == TransitionId::BeginRun) {
                        offsetReset(); // reset offset when writing out a new file
                        _writeDgram(reinterpret_cast<Dgram*>(m_configureBuffer.data()));
                    }
                    _writeDgram(const_cast<CubeResultDgram*>(&result));
                    if ((transitionId == TransitionId::Enable) && m_chunkRequest) {
                        logging::debug("%s calling reopenFiles()", __PRETTY_FUNCTION__);
                        reopenFiles();
                    } else if (transitionId == TransitionId::EndRun) {
                        //  Write all bins here?
                        logging::debug("%s calling closeFiles()", __PRETTY_FUNCTION__);
                        closeFiles();
                    }
                }
            }

            // Free the transition datagram buffer
            if (!dgram->isEvent()) {
                m_pool.freeTr(dgram);
            }
            
            // Free the pebble datagram buffer
            m_pool.freePebble(index);
        }
    }
    logging::warning("CubeTebReceiver::process is exiting");
}

/*
void collectorFunc(CubeTebReceiver& this)
{
     this.process();
}
*/

