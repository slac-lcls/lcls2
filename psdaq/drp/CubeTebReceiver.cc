/**
 **  This TebReceiver subclass accumulates all the Detector::rawDef data into bins as indicated
 **  by the CubeResultDgram returned from the TEB.  The data is organized as
 **     bins   : a uint32 array of bin indices
 **     entries: a uint32 array of accumulations per bin
 **     raw data: each field upconverted to double and rank increased by 1 with nbins at the first dimension
 **     (Note: the sizes of the raw data fields are known until L1A)
 **  The full nbins cube is recorded on EndRun.  Individual bins (the above structure with dim[0]=1
 **  are recorded and/or monitored as indicated by CubeResultDgram
 **/
#include "CubeTebReceiver.hh"
#include "CubeData.hh"

#include "psalg/utils/SysLog.hh"
#include "psdaq/service/kwargs.hh"
#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/Smd.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include "psdaq/eb/CubeResultDgram.hh"
#include "psdaq/eb/CubeConfigDgram.hh"
#include "psdaq/service/Json2Xtc.hh"

#include <sys/prctl.h>

using namespace XtcData;
using namespace Drp;
using namespace Pds;
using namespace Pds::Eb;
using logging = psalg::SysLog;
using us_t = std::chrono::microseconds;

typedef std::vector<ShapesData*> SDV;

//#define BIN_NORM   // Normalize

namespace Drp {
    //
    //  Find the ShapesData in the xtc.  This could be simpler with some assumptions.
    //
    class MyIterator : public XtcData::XtcIterator {
    public:
        MyIterator(std::vector<NamesId>& id) : 
            m_id(id),
            m_shapesdata(id.size(), 0) {}
    public:
        const SDV& shapesdata() const {
            return m_shapesdata;
        }

        int process(Xtc* xtc, const void* bufEnd)
        {
            switch(xtc->contains.id()) {
            case (TypeId::Parent): {
                iterate(xtc, bufEnd);
                break;
            }
            case (TypeId::ShapesData): {
                ShapesData& shapesdata = *(ShapesData*)xtc;
                NamesId& id = shapesdata.namesId();
                for(unsigned i=0; i<m_id.size(); i++) {
                    if (id==m_id[i]) {  // Found it!
                        m_shapesdata[i] = &shapesdata;
                        break;
                    }
                }
                break;
            }
            default:
                break; 
            }
            return 1;
        }
    private:
        std::vector<NamesId>&    m_id;
        SDV                      m_shapesdata;
    };

};


void subWorkerFunc(Detector&                   det,
                   SDV&                        rawShapesData,
                   unsigned                    nbins,
                   CubeData&                   cubeData,
                   SPSCQueue<unsigned>&        inputQueue,
                   Pds::Semaphore&             sem,
                   unsigned                    group,
                   unsigned                    subIndex)
{
    logging::info("CubeSubWorker %u.%u is starting with process ID %lu", group, subIndex, syscall(SYS_gettid));
    char nameBuf[16];
    snprintf(nameBuf, sizeof(nameBuf), "drp/CubeSub%d.%d", group, subIndex);
    if (prctl(PR_SET_NAME, nameBuf, 0, 0, 0) == -1) {
        perror("prctl");
    }

    //
    //  Event loop
    //
    while (true) {
        unsigned bin;
        if (!inputQueue.pop(bin)) [[unlikely]] {
                break;
            }

        cubeData.addSub(bin, subIndex, rawShapesData);
        sem.give();
    }

    logging::info("CubeSubWorker %u.%u is exiting", group, subIndex);
}

//
//  Each worker has independent memory allocated for the cube at Configure
//
void workerFunc(const Parameters& para, Detector& det, MemPool& pool,
                std::vector<CubeResultDgram>& results,
                unsigned nbins,
                std::atomic<bool>& data_init,
                CubeData& cubeData,
                Pds::Semaphore& sem,
                SPSCQueue<unsigned>& inputQueue, SPSCQueue<unsigned>& outputQueue,
                unsigned threadNum)
{
    unsigned  index;
    unsigned  next_index = threadNum;

    logging::info("CubeWorker %u is starting with process ID %lu", threadNum, syscall(SYS_gettid));
    char nameBuf[16];
    snprintf(nameBuf, sizeof(nameBuf), "drp/CubeWrk%d", threadNum);
    if (prctl(PR_SET_NAME, nameBuf, 0, 0, 0) == -1) {
        perror("prctl");
    }

    //
    //  Launch sub workers here
    //
    std::vector<SPSCQueue<unsigned> > subWorkerInputQueues;
    std::vector<Pds::Semaphore>       subWorkerSem;
    std::vector<std::thread>          subWorkerThreads;
    SDV                               rawShapesData;

    for (unsigned i=1; i<det.subIndices(); i++) {
        subWorkerInputQueues.emplace_back(SPSCQueue<unsigned>(1));
        subWorkerSem.emplace_back(Pds::Semaphore(Pds::Semaphore::EMPTY));
    }
    //  Sub workers need the rawshapesdata for each of the rawdef elements
    for (unsigned i=1; i<det.subIndices(); i++) {
        subWorkerThreads.emplace_back(subWorkerFunc,
                                      std::ref(det),
                                      std::ref(rawShapesData),
                                      nbins,
                                      std::ref(cubeData),
                                      std::ref(subWorkerInputQueues[i-1]),
                                      std::ref(subWorkerSem[i-1]),
                                      threadNum,
                                      i);
    }

    std::vector<VarDef>  rawDefV = det.rawDef();
    std::vector<NamesId> rawNames;
    for(unsigned i=0; i<rawDefV.size(); i++)
        rawNames.push_back( NamesId(det.nodeId, det.rawNamesIndex()+i) );

    //
    //  Event loop
    //
    while (true) {

        if (!inputQueue.pop(index)) [[unlikely]] {
                break;
            }

        const CubeResultDgram& result = results[index];
        TransitionId::Value transitionId = result.service();
        auto dgram = transitionId == TransitionId::L1Accept ? (EbDgram*)pool.pebble[index]
            : pool.transitionDgrams[index];

        if (index != next_index) {
            logging::error("Worker %u index %u != next %u [%u.%09u]", 
                           threadNum, index, next_index, 
                           dgram->time.seconds(), dgram->time.nanoseconds());
            abort();
        }
        next_index = (index + para.nCubeWorkers) % pool.nbuffers();

        // Event
        if (transitionId == TransitionId::L1Accept && result.persist()) {

            unsigned bin = result.binIndex();
            if (bin >= nbins) {
                logging::error("Bin index (%u) >= number of bins (%u)", bin, nbins);
                abort();
            }

            MyIterator iter(rawNames);
            iter.iterate(&dgram->xtc, dgram->xtc.next());

            //  Get the raw data (for size)
            rawShapesData = iter.shapesdata();

            if (data_init.load(std::memory_order_relaxed)) [[unlikely]] {

                if (rawDefV.size()==1 && iter.shapesdata()[0]==0) {
                    logging::warning("Skipping cube initialization due to missing data");
                    continue;
                }

                //
                //  Initialize the whole cube here
                //
                cubeData.initialize(dgram->xtc.src,
                                    iter.shapesdata(),
                                    threadNum);
                data_init.store(false, std::memory_order_release);
            }  // data_init
            {
                sem.take();

                //  Sub workers help here
                for(unsigned i=1; i<det.subIndices(); i++)
                    subWorkerInputQueues[i-1].push(bin);

                cubeData.add(bin, iter.shapesdata());

                for(unsigned i=1; i<det.subIndices(); i++)
                    subWorkerSem[i-1].take();

                sem.give();
            }
        }

        outputQueue.push(index);
    }

    //  Shut down sub workers
    for(unsigned i=1; i<det.subIndices(); i++) {
        subWorkerInputQueues[i-1].shutdown();
        if (subWorkerThreads[i-1].joinable())
            subWorkerThreads[i-1].join();
        logging::info("SubWorker %u.%u joined",threadNum,i);
    }

    logging::info("CubeWorker %u is exiting", threadNum);
}

CubeTebReceiver::CubeTebReceiver(const Parameters& para, DrpBase& drp) :
    TebReceiver  (para, drp),
    m_det        (drp.detector()),
    m_result     (m_pool.nbuffers(), CubeResultDgram(EbDgram(PulseId(0),Dgram()),0)),
    m_current    (-1),
    m_last       (-1),
    m_nbins      (0),
    m_data_init  (para.nCubeWorkers),
    m_buffer     (0),
    m_sem        (para.nCubeWorkers,Pds::Semaphore(Pds::Semaphore::FULL)),
    m_flush_sem  (Pds::Semaphore::EMPTY),
    m_terminate  (false)
{
}

//
//  Handles result
//    Queues binning summation to workers, if indicated
//
void CubeTebReceiver::complete(unsigned index, const ResultDgram& res)
{
    // This function is called by the base class's process() method to complete
    // processing and dispose of the event.  It presumes that the caller has
    // already vetted index and result
    const Pds::Eb::CubeResultDgram& result = reinterpret_cast<const Pds::Eb::CubeResultDgram&>(res);
    logging::debug("CubeTebReceiver::complete index (%u) result data(%x) persist(%u) monitor(%x) bin(%u) binMonitor(%u) binRecord(%u)", 
                   index, result.data(), result.persist(), result.monitor(), result.binIndex(), result.updateMonitor(), result.updateRecord());
    //    result.dump("complete");

    _queueDgram(index, result); // copies the result

    if (result.flush())
        m_flush_sem.take();
}

void CubeTebReceiver::_queueDgram(unsigned index, const Pds::Eb::CubeResultDgram& result)
{
    TransitionId::Value transitionId = result.service();
    if (transitionId != TransitionId::L1Accept) {
        auto dgram = m_pool.transitionDgrams[index];
        if (transitionId == TransitionId::Configure) {
            //  Nbins comes from result
            m_nbins = result.binIndex()+1;
            unsigned nbins = m_nbins;
            for(unsigned i=0; i<m_para.nCubeWorkers; i++)
                m_cubedata.push_back(new CubeData(m_det, nbins, m_pool.bufferSize()));

            //  Startup threads
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
                                             nbins,
                                             std::ref(m_data_init[i]),
                                             std::ref(*m_cubedata[i]),
                                             std::ref(m_sem[i]),
                                             std::ref(m_workerInputQueues[i]),
                                             std::ref(m_workerOutputQueues[i]),
                                             i);
            }
            m_terminate.store(false, std::memory_order_release);
            m_collectorThread = std::thread(&CubeTebReceiver::process, std::ref(*this));

            //  Write names for bin xtc
            const char* bufEnd = (char*)dgram + m_para.maxTrSize;
            m_cubedata[0]->addNames(m_para.detSegment, dgram, bufEnd);

            //  Let the timing system record the teb configuration data
            if (m_drp.nodeId() == m_tsId) {
                // convert to json to xtc
                char* json = result.xtc.payload()+sizeof(CubeConfigDgram)-sizeof(EbDgram);
                //size_t size = sizeof(*dgram) + dgram->xtc.sizeofPayload();

                logging::info("CubeResult Configure extent 0x%x  json %s", 
                              result.xtc.extent, json);

                auto config_buf = (char*)dgram->xtc.next();
                auto config_end = (char*)dgram + m_pool.pebble.trBufSize();
                NamesId configNamesId(m_det.nodeId,m_det.cubeNamesIndex()+1);
                unsigned len = Pds::translateJson2Xtc(json, config_buf, config_end, configNamesId, "cubeinfo", 0);
                dgram->xtc.alloc(len, config_end);
            }
        }
        else if (transitionId == TransitionId::Unconfigure) {
            //  Stop threads
            for(unsigned i=0; i<m_para.nCubeWorkers; i++) {
                m_workerInputQueues[i].shutdown();
                if (m_workerThreads[i].joinable())
                    m_workerThreads[i].join();
                logging::info("Worker %u joined",i);
            }
            for(unsigned i=0; i<m_para.nCubeWorkers; i++) {
                m_workerOutputQueues[i].shutdown();
            }

            m_terminate.store(true, std::memory_order_release);

            if (m_collectorThread.joinable()) {
                m_collectorThread.join();
                logging::info("CollectorThread joined");
            }

            m_workerInputQueues.clear();
            m_workerThreads.clear();
            m_workerOutputQueues.clear();

            for(unsigned i=0; i<m_cubedata.size(); i++)
                delete m_cubedata[i];
            m_cubedata.clear();

            // MEBs need the Unconfigure
            _monitorDgram(index, result);

            // Free the transition datagram buffer
            m_pool.freeTr(dgram);
            
            // Free the pebble datagram buffer
            m_pool.freePebble(index);

            m_current    = -1;
            m_last       = -1;
            return;
        }
        else if (transitionId == TransitionId::BeginRun) {
            //  Clear the cube every BeginRun
            for(unsigned i=0; i<m_para.nCubeWorkers; i++)
                m_data_init[i].store(true, std::memory_order_release);
        }
    }
    
    m_result[index] = result; // must copy (full xtc not copied!)
    
    unsigned worker = ++m_last % m_para.nCubeWorkers;
    if (index != (m_last % m_pool.nbuffers())) {
        logging::error("CubeTebReceiver::queueDgram index (%u) != m_last (%u) % nbuffers (%u)", index, m_last, m_pool.nbuffers());
        abort();
    }

    m_workerInputQueues[worker].push(index);
}

//
//  Copy one bin into dg
//    Requires summing over all workers
// 
Pds::EbDgram* CubeTebReceiver::_binDgram(Pds::EbDgram* dg, const CubeResultDgram& result)
{
    unsigned ibin = result.binIndex();
    SDV shapesDataV;

    //  Initialize and set shapes from the first worker
    unsigned iworker = 0;
    {
        //  Skip workers with no entries
        while (m_data_init[iworker].load(std::memory_order_relaxed)) {
            if (++iworker == m_para.nCubeWorkers) {
                //  There is no data to gather
                //  Do we need to do anything to indicate an empty dg?
                return dg;
            }
        }
        m_sem[iworker].take();
        dg = m_cubedata[iworker]->copyBin(ibin, shapesDataV, dg);
        m_sem[iworker].give();
    }

    //
    //  Accumulate from the rest of the workers
    //
    while(++iworker < m_para.nCubeWorkers) {
        //  Skip workers with no entries
        if (m_data_init[iworker].load(std::memory_order_relaxed))
            continue;

        m_sem[iworker].take();
        m_cubedata[iworker]->addBin(ibin, shapesDataV, dg);
        m_sem[iworker].give();
    }

    //    result.dump("binDgram");

    return dg;
}

void CubeTebReceiver::_monitorDgram(unsigned index, const CubeResultDgram& result)
{
    //    result.dump("monitorDgram");

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
                if (result.updateMonitor()) {
                    m_mon.post(_binDgram(dgram, result), result.monBufNo());
                }
                else {
                    m_mon.post(dgram, result.monBufNo());
                }
            }
        }
        // Other Transition
        else {
            m_mon.post(dgram);
        }
    }
}

void CubeTebReceiver::_recordDgram(unsigned index, const CubeResultDgram& result)
{
    TransitionId::Value transitionId = result.service();
    auto dgram = transitionId == TransitionId::L1Accept ? (EbDgram*)m_pool.pebble[index]
        : m_pool.transitionDgrams[index];

    // bool keepRaw = (m_drp.nodeId() == m_tsId);  // raw timing is always recorded
    bool keepRaw = m_para.cubeKeepRaw;

    if (writing()) {                    // Won't ever be true for Configure
        if (result.persist() || result.prescale()) {
            if (result.updateRecord()) {
                if (!keepRaw)           //  Write the intermediate accumulated bin (only)
                    dgram->xtc.extent = sizeof(Xtc);
                EbDgram* ebdg = _binDgram(dgram,result);
                _writeDgram(ebdg);
            }
            else {
                //  Do I need to record an empty dgram for the offline event builder?
                if (!keepRaw)
                    dgram->xtc.extent = sizeof(Xtc);
                _writeDgram(dgram);
            }
        }
        else if (transitionId != TransitionId::L1Accept) {

            if (transitionId == TransitionId::BeginRun) {
                offsetReset(); // reset offset when writing out a new file
                _writeDgram(reinterpret_cast<Dgram*>(m_configureBuffer.data()));
            }

            if (transitionId == TransitionId::EndRun) {

                unsigned iworker = 0;
                while (m_data_init[iworker].load(std::memory_order_relaxed)) {
                    if (++iworker == m_para.nCubeWorkers)
                        break;
                }

                if (iworker == m_para.nCubeWorkers) {
                    //  There is no data to gather
                    //  Do we need to do anything to indicate an empty dg?
                    logging::warning("Empty cube");
                    _writeDgram(dgram);
                }
                else {

                    std::vector<Dgram*> dg = m_cubedata[iworker]->dgram(dgram, transitionId);

                    while(++iworker < m_para.nCubeWorkers) {
                        if (m_data_init[iworker].load(std::memory_order_relaxed))
                            continue;
                        
                        m_sem[iworker].take();
                        m_cubedata[iworker]->add(dg);
                        m_sem[iworker].give();
                    }

                    for(auto& v : dg) {
                        if (chunkSize() > TebReceiverBase::DefaultChunkThresh) {
                            advanceChunkId();
                            reopenFiles();
                        }
                        _writeDgram(v);
                    }
                }
            }
            else {
                _writeDgram(dgram);
            }

            if ((transitionId == TransitionId::Enable) && m_chunkRequest) {
                logging::info("%s calling reopenFiles()", __PRETTY_FUNCTION__);
                reopenFiles();
            } else if (transitionId == TransitionId::EndRun) {
                logging::info("%s calling closeFiles()", __PRETTY_FUNCTION__);
                closeFiles();
            }
        }
    }
    else if (transitionId == TransitionId::Configure) {
        //  Update the configure cache (in TebReceiverBase)
        Dgram* configDgram = m_pool.transitionDgrams[index];
        size_t size = sizeof(*configDgram) + configDgram->xtc.sizeofPayload();
        memcpy(m_configureBuffer.data(), configDgram, size);
    }

}

void CubeTebReceiver::process()
{
    logging::info("CubeTebReceiver::process is starting with process ID %lu", syscall(SYS_gettid));
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

        unsigned index = ++m_current % m_pool.nbuffers();
        //  Need to make sure the worker is done with this buffer
        const CubeResultDgram& result = m_result[index];
        unsigned worker = m_current%m_para.nCubeWorkers;

        unsigned windex;
        bool rc = m_workerOutputQueues[worker].pop(windex);
        if (rc) {

            logging::debug("CubeTebReceiver::process index (%u) result(%x) bin (%u) worker (%u)",
                           index, result.data(), result.binIndex(), worker);

            if (windex != index) {
                logging::error("CubeTebReceiver::process index %u  windex %u",
                               index,windex);
                abort();
            }

            _monitorDgram(index, result);
            _recordDgram(index, result);

            // Free the transition datagram buffer
            TransitionId::Value transitionId = result.service();
            auto dgram = transitionId == TransitionId::L1Accept ? (EbDgram*)m_pool.pebble[index]
                : m_pool.transitionDgrams[index];
            if (!dgram->isEvent()) {
                m_pool.freeTr(dgram);
            }
            
            // Free the pebble datagram buffer
            m_pool.freePebble(index);

            if (result.flush()) {
                //  Clear the cube
                for(unsigned i=0; i<m_para.nCubeWorkers; i++)
                    m_data_init[i].store(true, std::memory_order_release);
                m_flush_sem.give();
            }
        }
        else {
            logging::info("CubeTebReceiver::process outputQ %u shutdown",worker);
            break;
        }
    }
    logging::info("CubeTebReceiver::process is exiting");
}
