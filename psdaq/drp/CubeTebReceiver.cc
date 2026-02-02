#include "CubeTebReceiver.hh"

#include "psalg/utils/SysLog.hh"
#include "psdaq/service/kwargs.hh"
#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/Smd.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include "psdaq/eb/CubeResultDgram.hh"

#include <sys/prctl.h>

using namespace XtcData;
using namespace Drp;
using namespace Pds;
using namespace Pds::Eb;
using logging = psalg::SysLog;
using us_t = std::chrono::microseconds;

#define BUFFER_SIZE(pool) pool.bufferSize()*10  // uint8_t -> double

namespace Drp {
    class CubeDef : public VarDef {
    public:
        enum { bin, entries };
        CubeDef(std::vector<Name>& detNames)
        {
            Alg cube("cube", 1, 0, 0);
            NameVec.push_back({"bin"    , Name::UINT32, 1, cube});
            NameVec.push_back({"entries", Name::UINT32, 1, cube});
            for(unsigned i=0; i<detNames.size(); i++)
                NameVec.push_back( { detNames[i].name(), detNames[i].type(), (int)detNames[i].rank()+1, cube} );
        }
    };

    class MyIterator : public XtcData::XtcIterator {
    public:
        MyIterator(NamesId& id) : 
            m_id(id),
            m_shapesdata(0) {}
    public:
        ShapesData* shapesdata() const {
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
                if (shapesdata.namesId()==m_id) {  // Found it!
                    m_shapesdata = &shapesdata;
                    return 0;
                }
                break;
            }
            default:
                break; 
            }
            return 1;
        }
    private:
        NamesId     m_id;
        ShapesData* m_shapesdata;
    };


    class DumpIterator : public XtcData::XtcIterator {
    public:
        DumpIterator(char* root, unsigned indent=0) : 
            m_root(root), m_indent(indent) {}
    public:
        int process(Xtc* xtc, const void* bufEnd)
        {
            printf("   [%u] xtc 0x%lx  typeid 0x%x  src 0x%x  extent 0x%x\n",
                   m_indent, (char*)xtc - m_root, xtc->contains.value(), xtc->src.value(), xtc->extent);
            switch(xtc->contains.id()) {
            case (TypeId::Parent):
            case (TypeId::ShapesData): {
                DumpIterator iter(m_root, m_indent+1);
                iter.iterate(xtc, bufEnd);
                break;
            }
            default:
                break; 
            }
            return 1;
        }
    private:
        char*    m_root;
        unsigned m_indent;
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
    unsigned  index;

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

        logging::debug("CubeWorker %u %s index (%u)",
                       threadNum, TransitionId::name(transitionId), index);
        // Event
        if (transitionId == TransitionId::L1Accept && result.persist()) {

            //NamesId rawNames = NamesId(det.nodeId,RawNamesIndex);
            NamesId rawNames = NamesId(det.nodeId,NamesIndex::BASE);
            MyIterator iter(rawNames);
            iter.iterate(&dgram->xtc, dgram->xtc.next());
            if (!iter.shapesdata()) {
                logging::error("Unable to find raw data in xtc");
                return;
            }

            unsigned ibin = result.bin();
            if (bin_entries[ibin]==0) {
                const char* bufEnd = bin_data[ibin]+BUFFER_SIZE(pool);
                Xtc& xtc = *new (bin_data[ibin], bufEnd) Xtc(TypeId(TypeId::Parent,0),dgram->xtc.src);
                det.cubeInit(*iter.shapesdata(), // raw shapes data
                             xtc,                // cube xtc
                             bufEnd);
                bin_entries[ibin]++;
            }
            else {
                logging::debug("CubeWorker bin (%u) entries (%u)", ibin, bin_entries[ibin]);
                Xtc& xtc = *(Xtc*)bin_data[ibin];
                sem.take();  // not bin-specific
                // add the event into the bin
                // the bin storage has to contain shape, thus the shapesdata;
                det.cubeAdd(*iter.shapesdata(), *(ShapesData*)xtc.payload());
                bin_entries[ibin]++;
                sem.give();
            }
        }

        outputQueue.push(index);
    }

    logging::info("CubeWorker %u is exiting", threadNum);
}

CubeTebReceiver::CubeTebReceiver(const Parameters& para, DrpBase& drp) :
    TebReceiver  (para, drp),
    m_det        (drp.detector()),
    m_result     (m_pool.nbuffers(), CubeResultDgram(EbDgram(PulseId(0),Dgram()),0)),
    m_current    (-1),
    m_last       (-1),
    m_bin_data   (MAX_CUBE_BINS+1),
    m_bin_entries(para.nCubeWorkers, std::vector<unsigned>(MAX_CUBE_BINS,0)),
    m_buffer     (new char[BUFFER_SIZE(drp.pool)]),
    m_sem        (para.nCubeWorkers,Pds::Semaphore(Pds::Semaphore::FULL)),
    m_terminate  (false)
{
     for(unsigned i=0; i<MAX_CUBE_BINS+1; i++)
         m_bin_data[i] = new char[BUFFER_SIZE(drp.pool)];
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
    logging::debug("CubeTebReceiver::complete index (%u) result data(%x) persist(%u) monitor(%x) bin(%u) worker(%u) record(%u)", 
                   index, result.data(), result.persist(), result.monitor(), result.bin(), result.worker(), result.record());

#if 0
    _monitorDgram(index, result);
#endif
    _queueDgram(index, result); // copies the result
}

void CubeTebReceiver::_queueDgram(unsigned index, const Pds::Eb::CubeResultDgram& result)
{
    TransitionId::Value transitionId = result.service();
    if (transitionId != TransitionId::L1Accept) {
        auto dgram = m_pool.transitionDgrams[index];
        if (transitionId == TransitionId::Configure) {
            //  Startup threads
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

            const char* bufEnd = (char*)dgram + m_para.maxTrSize;

            { DumpIterator dump((char*)&dgram->xtc);
                dump.iterate(&dgram->xtc, bufEnd); }

            //  Write names for bin xtc
            Alg cubeAlg("cube", 2, 0, 0);
            NamesId cubeNamesId(m_det.nodeId,m_det.cubeNamesIndex());
            Names& cubeNames = *new(dgram->xtc, bufEnd)
                Names(bufEnd,
                      m_para.detName.c_str(), cubeAlg,
                      m_para.detType.c_str(), m_para.serNo.c_str(), cubeNamesId, m_para.detSegment);
            std::vector<Name> nameVec(m_det.cubeDef().NameVec);
            CubeDef myCubeDef(nameVec);
            cubeNames.add(dgram->xtc, bufEnd, myCubeDef);
            m_namesLookup[cubeNamesId] = NameIndex(cubeNames);

            { DumpIterator dump((char*)&dgram->xtc);
                dump.iterate(&dgram->xtc, bufEnd); }
        }
        else if (transitionId == TransitionId::Unconfigure) {
            //  Stop threads
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

            // MEBs need the Unconfigure
            _monitorDgram(index, result);

            // Free the transition datagram buffer
            m_pool.freeTr(dgram);
            
            // Free the pebble datagram buffer
            m_pool.freePebble(index);
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

static void _dumpNames(DescData& data, const char* title)
{
    Names& names = data.nameindex().names();
    logging::debug("Found %d names in %s",names.num(), title);
    for(unsigned i=0; i<names.num(); i++) {
        Name& name = names.get(i);
        logging::debug("\t%d : %s (%s)[%u]",i,name.name(),name.str_type(),name.rank());
    }
}

Pds::EbDgram* CubeTebReceiver::_binDgram(Pds::EbDgram* dg, const CubeResultDgram& result)
{
    //  Record one bin
    unsigned ibin = result.bin();
    uint32_t scalar_array[] = {1,0,0,0,0};
    Shape    scalar(scalar_array);

    NamesId namesId(m_det.nodeId, m_det.cubeNamesIndex());
    //  DescribedData creates the Data container first, then the Shapes container
    DescribedData data(dg->xtc, m_buffer+BUFFER_SIZE(m_pool), m_namesLookup, namesId);
    _dumpNames(data, "binDgram");

    //  First two entries come from us
    ((uint32_t*)data.data())[0] = ibin;
    ((uint32_t*)data.data())[1] = m_bin_entries[result.worker()][ibin];
    unsigned size = 2*sizeof(uint32_t);

    //  Remainder of entries come from the bin data given by Detector
    Xtc&        xtc = *(Xtc*)m_bin_data[ibin];
    ShapesData& shd = *(ShapesData*)xtc.payload();
    Shapes&      sh = shd.shapes();
    Data&        da = shd.data();
    memcpy((uint8_t*)data.data() + size, da.payload(), da.sizeofPayload());
    size += da.sizeofPayload();
    data.set_data_length(size);

    //  By DescribedData, we can't set the shapes until all the data is allocated
    data.set_array_shape(0, scalar.shape());
    data.set_array_shape(1, scalar.shape());
    unsigned iarray=0;
    VarDef  cubeDef = m_det.cubeDef();
    for(unsigned i=0; i<cubeDef.NameVec.size(); i++) {
        unsigned rank = cubeDef.NameVec[i].rank();
        Shape s(scalar);
        if (rank) {
            s = Shape(sh.get(iarray++));
            s.shape()[rank] = 1;
        }
        data.set_array_shape(i+2, s.shape());
    }
    
    logging::info("binDgram bin (%u) worker (%u) pulseId (%lu/0x%016lx) timeStamp (%lu/0x%016lx) size (%u)",
                   result.bin(), result.worker(), 
                   result.pulseId(), result.pulseId(), 
                   result.time.value(), result.time.value(), 
                   size);
    for(unsigned i=0; i<cubeDef.NameVec.size()+2; i++) {
        const uint32_t* s = data.shapesdata().shapes().get(i).shape();
        logging::info("shape %u: %u %u %u %u %u",
                      i, s[0], s[1], s[2], s[3], s[4]);
    }
    { DumpIterator dump((char*)&xtc);
        dump.iterate(&xtc, m_bin_data[ibin]+BUFFER_SIZE(m_pool)); }

    { DumpIterator dump((char*)dg);
        dump.iterate(&dg->xtc, m_buffer+BUFFER_SIZE(m_pool)); }

    return dg;
}

void CubeTebReceiver::_monitorDgram(unsigned index, const CubeResultDgram& result)
{
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
                if (result.record()) {
                    //  Write the intermediate accumulated bin
                    m_sem[result.worker()].take();
                    memcpy(m_buffer, dgram, sizeof(*dgram)+dgram->xtc.sizeofPayload());
                    m_mon.post(_binDgram((Pds::EbDgram*)m_buffer, result), result.monBufNo());
                    m_sem[result.worker()].give();
                }
                else {
                    m_mon.post(dgram, result.monBufNo());
                    logging::info("Standard L1");
                    DumpIterator iter((char*)dgram);
                    iter.iterate(&dgram->xtc, (char*)dgram + m_pool.bufferSize());
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

    if (writing()) {                    // Won't ever be true for Configure
        if (result.persist() || result.prescale()) {
            if (result.record()) {
                //  Write the intermediate accumulated bin
                m_sem[result.worker()].take();
                Pds::EbDgram* dg = new(m_buffer) Pds::EbDgram(result, result);  // PulseId, Dgram
                _writeDgram(_binDgram(dg, result));
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
                logging::info("%s calling reopenFiles()", __PRETTY_FUNCTION__);
                reopenFiles();
            } else if (transitionId == TransitionId::EndRun) {
                //  Write all bins here?
                logging::info("%s calling closeFiles()", __PRETTY_FUNCTION__);
                closeFiles();
            }
        }
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

        m_current = (m_current + 1) % (m_pool.nbuffers() - 1);

        unsigned index = m_current;
        //  Need to make sure the worker is done with this buffer
        const CubeResultDgram& result = m_result[index];

        logging::debug("CubeTebReceiver::process index (%u) result(%x) bin (%u) worker (%u)",
                       index, result.data(), result.bin(), result.worker());

        unsigned windex;
        bool rc = m_workerOutputQueues[result.worker()].pop(windex);
        if (rc) {

            logging::debug("CubeTebReceiver::process index %u  windex %u",
                           index,windex);

            _recordDgram(index, result);
#if 1
            _monitorDgram(index, result);
#endif
            // Free the transition datagram buffer
            TransitionId::Value transitionId = result.service();
            auto dgram = transitionId == TransitionId::L1Accept ? (EbDgram*)m_pool.pebble[index]
                : m_pool.transitionDgrams[index];
            if (!dgram->isEvent()) {
                m_pool.freeTr(dgram);
            }
            
            // Free the pebble datagram buffer
            m_pool.freePebble(index);
        }
    }
    logging::info("CubeTebReceiver::process is exiting");
}
