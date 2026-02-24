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

typedef std::vector<ShapesData*> SDV;

//#define DBUG
//#define BIN_NORM   // Normalize

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
                NameVec.push_back( { detNames[i].name(), Name::DOUBLE, (int)detNames[i].rank()+1, cube} );
        }
    };

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


#define DUMP_XTC(title,xtc,base) {                                      \
        uint32_t* p = (uint32_t*)xtc;                                   \
        printf("%s xtc [%lx] %08x %08x %08x  extent %x\n",              \
               title, (char*)xtc - (char*)base, p[0], p[1], p[2], xtc->extent); }

#define DUMP_DGRAM(title,dgram) {                                     \
        uint32_t* p = (uint32_t*)dgram;                               \
        printf("[%s] dg %08x %08x %08x %08x %08x  size %x\n",         \
               title, p[0], p[1], p[2], p[3], p[4], dgram->xtc.sizeofPayload()); }

void subWorkerFunc(Detector&                   det,
                   SDV&                        rawShapesData,
                   unsigned                    nbins,
                   char*                       bin_data,
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

    std::vector<VarDef>& rawDefV = det.rawDef();

    //
    //  Event loop
    //
    while (true) {
        unsigned bin;
        if (!inputQueue.pop(bin)) [[unlikely]] {
                break;
            }

        SDV rsd =  rawShapesData;

        Xtc* xtc = (Xtc*)(((Dgram*)bin_data)->xtc.payload());

        for(unsigned idef=0; idef<rawDefV.size(); idef++) {
            NamesId rawNamesId (det.nodeId,det.rawNamesIndex ()+idef);
            NamesId cubeNamesId(det.nodeId,det.cubeNamesIndex()+idef);
            DescData rawdata(*rsd[idef], det.namesLookup()[rawNamesId]);  // reading

            DescData cubedata(*(ShapesData*)xtc, det.namesLookup()[cubeNamesId]);

            unsigned size = 2*nbins*sizeof(uint32_t);

            VarDef& rawDef = rawDefV[idef];
            for(unsigned i=0; i<rawDef.NameVec.size(); i++) {
                Name& name = rawDef.NameVec[i];
                unsigned arraySize = name.rank() ? sizeof(double_t)*Shape(rawdata.shape(name)).num_elements(name.rank()) : sizeof(double_t);
                double* dst = (double_t*)((char*)cubedata.shapesdata().data().payload()+size+bin*arraySize);
                size += nbins*arraySize;
                det.addToCube(idef, i, subIndex, dst, rawdata);
            }
            xtc = (Xtc*)(xtc->next());
        }
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
                char* bin_data,
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
                                      std::ref(bin_data),
                                      std::ref(subWorkerInputQueues[i-1]),
                                      std::ref(subWorkerSem[i-1]),
                                      threadNum,
                                      i);
    }

    std::vector<VarDef> rawDefV = det.rawDef();
    std::vector<VarDef> cubeDefV;
    for(auto& v : rawDefV)
        cubeDefV.push_back(CubeDef(v.NameVec));

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

            std::vector<NamesId> rawNames;
            for(unsigned i=0; i<rawDefV.size(); i++)
                rawNames.push_back( NamesId(det.nodeId, det.rawNamesIndex()+i) );

            MyIterator iter(rawNames);
            iter.iterate(&dgram->xtc, dgram->xtc.next());

            //  Get the raw data (for size)
            rawShapesData = iter.shapesdata();

#ifdef DBUG
            for(unsigned i=0; i<rawDefV.size(); i++)
                logging::info(" rawShapesData[%i] = %p",i,rawShapesData[i]);
#endif

            if (data_init.load(std::memory_order_relaxed)) [[unlikely]] {

                if (rawDefV.size()==1 && iter.shapesdata()[0]==0) {
                    logging::warning("Skipping cube initialization due to missing data");
                    continue;
                }

                //
                //  Initialize the whole cube here
                //
                const char* bufEnd = bin_data+nbins*det.cubeBinBytes();
                Dgram& dg = *new (bin_data) Dgram( Transition(), Xtc(TypeId(TypeId::Parent,0),dgram->xtc.src) );
                Xtc& xtc = dg.xtc;
                
                {   printf("data_init before\n");
                    DumpIterator dump((char*)&xtc);
                    dump.iterate(&xtc, bufEnd); }
                
                for(unsigned idef=0; idef<rawDefV.size(); idef++) {

                    //  I can't really skip initialization here
                    DescData* rawdata = iter.shapesdata()[idef] ?
                        new DescData(*iter.shapesdata()[idef], 
                                     det.namesLookup()[rawNames[idef]] ) : 0;

                    VarDef& rawDef = rawDefV[idef];
                    VarDef& cubeDef = cubeDefV[idef];

                    NamesId namesId(det.nodeId, det.cubeNamesIndex()+idef);
                    //  DescribedData creates the Data container first, then the Shapes container
                    DescribedData data(xtc, bufEnd, det.namesLookup(), namesId);
                    //  Fill the data payload first
                    //  bin indices and entries (same for all detectors)
                    unsigned size=0;
                    {
                        uint32_t* p = (uint32_t*)data.data();
                        for(unsigned i=0; i<nbins; i++)  // bin indices
                            p[i] = i;
                        memset(&p[nbins], 0, nbins*sizeof(uint32_t));  // bin entries
                        size += 2*nbins*sizeof(uint32_t);
                    }

                    //  Detector-specific payload
                    //  Everything here becomes double
                    for(unsigned i=0; i<rawDef.NameVec.size(); i++) {
                        unsigned rank = rawDef.NameVec[i].rank();
                        double_t* dst = (double_t*)((char*)data.data()+size);
                        if (rank==0) {
                            memset(dst, 0, nbins*sizeof(double_t));
                            size += nbins*sizeof(double_t);
                        }
                        else {
                            if (rawdata==0) {
                                logging::error("Initializing cube vector without raw data");
                                abort();
                            }
                            uint32_t newShape[XtcData::MaxRank];
                            newShape[0] = nbins;
                            for(unsigned r=0; r<4; r++)
                                newShape[r+1] = rawdata->shape(rawDef.NameVec[i])[r];
                            Shape s(newShape);
                            unsigned arraySize = s.size(cubeDef.NameVec[i+2]); 
                            memset(dst, 0, arraySize);
                            size += arraySize;
                        }
                        if (threadNum==0)
                            logging::info("cube: %s dst %p size 0x%lx",rawDef.NameVec[i].name(),dst, size);
                    }
                    data.set_data_length(size);

                    //
                    //  Now set the shapes
                    //
                    uint32_t scalar_array[] = {nbins,0,0,0,0};
                    Shape scalar(scalar_array);
                    data.set_array_shape(CubeDef::bin    , scalar.shape());
                    data.set_array_shape(CubeDef::entries, scalar.shape());
                    for(unsigned i=2; i<cubeDef.NameVec.size(); i++) {
                        unsigned rank = cubeDef.NameVec[i].rank();
                        Shape s(scalar_array);
                        if (rank>1) {
                            uint32_t newShape[XtcData::MaxRank];
                            newShape[0] = nbins;
                            for(unsigned r=0; r<4; r++)
                                newShape[r+1] = rawdata->shape(rawDef.NameVec[i-2])[r];
                            s = Shape(newShape);
                        }
                        data.set_array_shape(i, s.shape());
                    }

                    //  Printout
                    if (threadNum==0) {
                        logging::info("Initialized cube with bin %u of %u bins.", bin, nbins);
                        ShapesData& shd = data.shapesdata();
                        logging::info("Shapes at %lx,  Data at %lx",
                                       (char*)&shd.shapes()-xtc.payload(), 
                                       (char*)&shd.data  ()-xtc.payload());
                        for(unsigned i=0; i<cubeDef.NameVec.size(); i++) {
                            uint32_t* sh = shd.shapes().get(i).shape();
                            logging::info("Shape[%u] (%s): %u %u %u %u %u",
                                           i, cubeDef.NameVec[i].name(), sh[0], sh[1], sh[2], sh[3], sh[4]);
                        } 
                    }
                } // for(idef...

                {   printf("data_init after\n");
                    DumpIterator dump((char*)&dg.xtc);
                    dump.iterate(&dg.xtc, bufEnd); }
                

                data_init.store(false, std::memory_order_release);
            }  // data_init
            {
                sem.take();

                //  Sub workers help here
                for(unsigned i=1; i<det.subIndices(); i++)
                    subWorkerInputQueues[i-1].push(bin);

                Xtc* xtc = (Xtc*)(((Dgram*)bin_data)->xtc.payload());

                for(unsigned idef=0; idef<rawDefV.size(); idef++) {

                    if (iter.shapesdata()[idef]) {
                        VarDef& rawDef = rawDefV[idef];

#ifdef DBUG
                        ShapesData* shpd = iter.shapesdata()[idef];
                        printf("idef %u  shapesdata %p\n", idef, shpd);
                        DUMP_XTC("SHPD", shpd             , (&dgram->xtc));
                        DUMP_XTC("DATA", (&shpd->data())  , (&dgram->xtc));
                        //DUMP_XTC("SHA",  (&shpd->shapes()), (&dgram->xtc)); // No shapes if no arrays
#endif
                        DescData rawdata(*(iter.shapesdata())[idef], 
                                         det.namesLookup()[rawNames[idef]] );

                        NamesId cubeNamesId(det.nodeId,det.cubeNamesIndex()+idef);
                        DescData cubedata(*(ShapesData*)xtc, det.namesLookup()[cubeNamesId]);
                        {
                            uint32_t* p = (uint32_t*)cubedata.shapesdata().data().payload();
                            p[nbins+bin]++;  // increment bin entries
                        }
                        unsigned size = 2*nbins*sizeof(uint32_t);
                    
                        for(unsigned i=0; i<rawDef.NameVec.size(); i++) {
                            Name& name = rawDef.NameVec[i];
                            unsigned arraySize = name.rank() ? sizeof(double_t)*Shape(rawdata.shape(name)).num_elements(name.rank()) : sizeof(double_t);
                            double* dst = (double_t*)((char*)cubedata.shapesdata().data().payload()+size+bin*arraySize);
                            size += nbins*arraySize;
                            det.addToCube(idef, i, 0, dst, rawdata);
                        }
                    }
                    
                    xtc = xtc->next();
                }

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
    m_bin_data   (para.nCubeWorkers),
    m_buffer     (new char[drp.detector().cubeBinBytes()]),
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
            logging::info("Allocating cube memory for %u bins", nbins);
            logging::info("CubeTebReceiver workers %u  event_buffer_size %lu  cube_buffer_size %lu",
                          m_para.nCubeWorkers, m_det.cubeBinBytes(), nbins*m_det.cubeBinBytes());
            for(unsigned i=0; i<m_para.nCubeWorkers; i++)
                m_bin_data[i] = new char[nbins*m_det.cubeBinBytes()];

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
                                             std::ref(m_bin_data[i]),
                                             std::ref(m_sem[i]),
                                             std::ref(m_workerInputQueues[i]),
                                             std::ref(m_workerOutputQueues[i]),
                                             i);
            }
            m_terminate.store(false, std::memory_order_release);
            m_collectorThread = std::thread(&CubeTebReceiver::process, std::ref(*this));

            const char* bufEnd = (char*)dgram + m_para.maxTrSize;

            {   printf("Names before\n");
                DumpIterator dump((char*)&dgram->xtc);
                dump.iterate(&dgram->xtc, bufEnd); }

            //  Write names for bin xtc
            Alg cubeAlg("cube", 2, 0, 0);
            std::vector<VarDef>& rawDefV = m_det.rawDef();
            for(unsigned i=0; i<rawDefV.size(); i++) {
                NamesId rawNamesId (m_det.nodeId,m_det.rawNamesIndex()+i);
                Names& rawNames = m_det.namesLookup()[rawNamesId].names();

                NamesId cubeNamesId(m_det.nodeId,m_det.cubeNamesIndex()+i);
                Names& cubeNames = *new(dgram->xtc, bufEnd)
                    Names(bufEnd,
                          rawNames.detName(), cubeAlg,
                          rawNames.detType(), rawNames.detId(), cubeNamesId, m_para.detSegment);
                VarDef& rawDef = rawDefV[i];
                VarDef cubeDef = CubeDef(rawDef.NameVec);
                cubeNames.add(dgram->xtc, bufEnd, cubeDef);
                m_det.namesLookup()[cubeNamesId] = NameIndex(cubeNames);
            }

            {   printf("Names after\n");
                DumpIterator dump((char*)&dgram->xtc);
                dump.iterate(&dgram->xtc, bufEnd); }
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

            for(unsigned i=0; i<m_para.nCubeWorkers; i++)
                delete[] m_bin_data[i];

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
    unsigned nbins = m_nbins;
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

#ifdef DBUG
        DUMP_DGRAM("TGT DG IN",dg);
#endif

        Xtc* bxtc = (Xtc*)((Dgram*)m_bin_data[iworker])->xtc.payload();

        std::vector<VarDef>& rawDefV = m_det.rawDef();

        for(unsigned idef=0; idef < rawDefV.size(); idef++) {
#ifdef DBUG
            DUMP_XTC("TGT",(&dg->xtc),dg);
            DUMP_XTC("SRC",bxtc,m_bin_data[iworker]);
#endif
            VarDef&rawDef  = rawDefV[idef];
            VarDef cubeDef = CubeDef(rawDef.NameVec);

            NamesId namesId(m_det.nodeId, m_det.cubeNamesIndex()+idef);
            //  DescribedData creates the Data container first, then the Shapes container
            DescribedData data(dg->xtc, m_buffer+m_det.cubeBinBytes(), m_det.namesLookup(), namesId);
            shapesDataV.push_back(&data.shapesdata());

            DescData cubedata(*(ShapesData*)bxtc, m_det.namesLookup()[namesId]);
            ((uint32_t*)data.data())[0] = ibin;
            ((uint32_t*)data.data())[1] = ((uint32_t*)cubedata.shapesdata().data().payload())[nbins+ibin]; // entries
            unsigned dstSize = 2*sizeof(uint32_t);
            unsigned srcSize = 2*sizeof(uint32_t)*nbins;

            //  The rest are double arrays
            for(unsigned i=2; i<cubeDef.NameVec.size(); i++) {
                Shape s(cubedata.shape(cubeDef.NameVec[i]));
                s.shape()[0] = 1;
                unsigned binSize = s.size(cubeDef.NameVec[i]);
                double_t* dst = (double_t*)((char*)data.data()+dstSize);
                double_t* src = (double_t*)((char*)cubedata.shapesdata().data().payload()+srcSize+ibin*binSize);
                memcpy(dst, src, binSize);
                dstSize += binSize;
                srcSize += binSize*nbins;
            }
            data.set_data_length(dstSize);

            //  Now set the shapes
            uint32_t scalar_array[] = {1,0,0,0,0};
            Shape    scalar(scalar_array);
            data.set_array_shape(CubeDef::bin    , scalar_array);
            data.set_array_shape(CubeDef::entries, scalar_array);
            for(unsigned i=2; i<cubeDef.NameVec.size(); i++) {
                Shape s(cubedata.shape(cubeDef.NameVec[i]));
                s.shape()[0] = 1;
                data.set_array_shape(i, s.shape());
            }

            bxtc = (Xtc*)bxtc->next();
        }
#ifdef DBUG
        DUMP_DGRAM("TGT DG OUT",dg);
        DUMP_XTC("TGT OUT",(&dg->xtc),dg);
#endif
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

        Xtc& bxtc = ((Dgram*)m_bin_data[iworker])->xtc;

        std::vector<VarDef>& rawDefV = m_det.rawDef();
        for(unsigned idef=0; idef < rawDefV.size(); idef++) {
            VarDef& rawDef = rawDefV[idef];
            VarDef cubeDef = CubeDef(rawDef.NameVec);

            NamesId namesId(m_det.nodeId, m_det.cubeNamesIndex()+idef);
            DescData cubedata(*(ShapesData*)(bxtc.payload()), m_det.namesLookup()[namesId]);
            void* data = shapesDataV[idef]->data().payload();
            ((uint32_t*)data)[1] += ((uint32_t*)(cubedata.shapesdata().data().payload()))[nbins+ibin]; // entries
            unsigned dstSize = 2*sizeof(uint32_t);
            unsigned srcSize = 2*sizeof(uint32_t)*nbins;

            //  The rest are double arrays
            for(unsigned i=2; i<cubeDef.NameVec.size(); i++) {
                Shape s(cubedata.shape(cubeDef.NameVec[i]));
                s.shape()[0] = 1;
                unsigned binSize = s.size(cubeDef.NameVec[i]);
                double_t* dst = (double_t*)((char*)data+dstSize);
                double_t* src = (double_t*)((char*)cubedata.shapesdata().data().payload()+srcSize+ibin*binSize);
                
                for(unsigned j=0; j<binSize/sizeof(double_t); j++)
                    dst[j] += src[j];

                dstSize += binSize;
                srcSize += binSize*nbins;
            }

            bxtc = *(bxtc.next());
        }
        m_sem[iworker].give();
    }

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
                if (result.updateMonitor()) {
                    //  Append the intermediate bin sum to the pebble data
                    memset(m_buffer, 0, m_det.cubeBinBytes()); // clear it for sure!!
                    memcpy(m_buffer, dgram, sizeof(*dgram)+dgram->xtc.sizeofPayload());
                    m_mon.post(_binDgram((Pds::EbDgram*)m_buffer, result), result.monBufNo());
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
    auto dgram = transitionId == TransitionId::L1Accept ? (EbDgram*)m_pool.pebble[index]
        : m_pool.transitionDgrams[index];

    if (writing()) {                    // Won't ever be true for Configure
        if (result.persist() || result.prescale()) {
            if (result.updateRecord()) {
                //  Write the intermediate accumulated bin
                Pds::EbDgram* dg = new(m_buffer) Pds::EbDgram(*dgram, *dgram);  // PulseId, Dgram
                _writeDgram(_binDgram(dg, result));
            }
            else {
                //  Do I need to record an empty dgram for the offline event builder?
                Pds::EbDgram* dg = new(m_buffer) Pds::EbDgram(*dgram, *dgram);  // PulseId, Dgram
                _writeDgram(dg);
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

                    unsigned nbins = m_nbins;
                    { Dgram* bindg = (Dgram*)m_bin_data[iworker];
                        DumpIterator dump((char*)bindg);
                        dump.iterate(&bindg->xtc, m_bin_data[iworker]+nbins*m_det.cubeBinBytes()); }

                    Dgram* dg = (Dgram*)m_bin_data[iworker];
#ifdef DBUG
                    DUMP_DGRAM("TGT0 DG IN",dg);
                    DUMP_XTC("TGT0",(&dg->xtc),dg);
#endif

                    //  Overwrite the header
                    new(m_bin_data[iworker]) Transition(dgram->type(), transitionId, dgram->time, dgram->env);

#ifdef DBUG
                    DUMP_DGRAM("TGT DG IN",dg);
#endif

                    std::vector<VarDef>& rawDefV = m_det.rawDef();

                    while(++iworker < m_para.nCubeWorkers) {

                        if (m_data_init[iworker].load(std::memory_order_relaxed))
                            continue;
                        
                        m_sem[iworker].take();

                        Xtc* xtc = (Xtc*)dg->xtc.payload();
                        Xtc* bxtc = (Xtc*)((Dgram*)m_bin_data[iworker])->xtc.payload();

                        for(unsigned idef=0; idef < rawDefV.size(); idef++) {
#ifdef DBUG
                            DUMP_XTC("TGT",xtc,dg);
                            DUMP_XTC("SRC",bxtc,m_bin_data[iworker]);
#endif
                            VarDef& rawDef = rawDefV[idef];
                            VarDef cubeDef = CubeDef(rawDef.NameVec);
                        
                            NamesId namesId(m_det.nodeId, m_det.cubeNamesIndex()+idef);
                            //  Sum the data
                            void* data = ((ShapesData*)xtc)->data().payload();
                            DescData cubedata(*(ShapesData*)bxtc, m_det.namesLookup()[namesId]);

                            // entries
                            for(unsigned bin=0; bin<nbins; bin++)
                                ((uint32_t*)data)[nbins+bin] += ((uint32_t*)(cubedata.shapesdata().data().payload()))[nbins+bin];
                            unsigned dstSize = 2*sizeof(uint32_t)*nbins;
                            unsigned srcSize = 2*sizeof(uint32_t)*nbins;

                            //  The rest are double arrays
                            for(unsigned i=2; i<cubeDef.NameVec.size(); i++) {
                                Shape s(cubedata.shape(cubeDef.NameVec[i]));
                                unsigned allSize = s.size(cubeDef.NameVec[i]);
                                double_t* dst = (double_t*)((char*)data+dstSize);
                                double_t* src = (double_t*)((char*)cubedata.shapesdata().data().payload()+srcSize);
                                // all bins
                                for(unsigned j=0; j<allSize/sizeof(double_t); j++)
                                    dst[j] += src[j];

                                dstSize += allSize;
                                srcSize += allSize;
                            }
                            xtc = xtc->next();
                            bxtc = bxtc->next();
                        }
#ifdef DBUG
                        DUMP_DGRAM("TGT DG OUT",dg);
                        DUMP_XTC("TGT OUT",(&dg->xtc),dg);
#endif
                        m_sem[iworker].give();
                    }

                    //  Until we get this figured out
                    _writeDgram(dg);
                    //_writeDgram(dgram);
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

            _recordDgram(index, result);
            _monitorDgram(index, result);

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
