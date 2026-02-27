#include "CubeData.hh"
#include "Detector.hh"

#include "psalg/utils/SysLog.hh"
#include "psdaq/service/EbDgram.hh"

#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/NameIndex.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/VarDef.hh"

namespace Drp {
    class CubeDef : public XtcData::VarDef {
    public:
        enum { bin, entries };
        CubeDef(std::vector<XtcData::Name>& detNames);
    };

    class DumpIterator : public XtcData::XtcIterator {
    public:
        DumpIterator(char* root, unsigned indent=0) : 
            m_root(root), m_indent(indent) {}
    public:
        int process(XtcData::Xtc* xtc, const void* bufEnd);
    private:
        char*    m_root;
        unsigned m_indent;
    };
}

using namespace Drp;
using namespace XtcData;
using namespace Pds;
using namespace Pds::Eb;
using logging = psalg::SysLog;

#define DUMP_XTC(title,xtc,base) {                                      \
        uint32_t* p = (uint32_t*)xtc;                                   \
        printf("%s xtc [%lx] %08x %08x %08x  extent %x\n",              \
               title, (char*)xtc - (char*)base, p[0], p[1], p[2], xtc->extent); }

#define DUMP_DGRAM(title,dgram) {                                     \
        uint32_t* p = (uint32_t*)dgram;                               \
        printf("[%s] dg %08x %08x %08x %08x %08x  size %x\n",         \
               title, p[0], p[1], p[2], p[3], p[4], dgram->xtc.sizeofPayload()); }


CubeDef::CubeDef(std::vector<Name>& detNames)
{
    Alg cube("cube", 1, 0, 0);
    NameVec.push_back({"bin"    , Name::UINT32, 1, cube});
    NameVec.push_back({"entries", Name::UINT32, 1, cube});
    for(unsigned i=0; i<detNames.size(); i++)
        NameVec.push_back( { detNames[i].name(), Name::DOUBLE, (int)detNames[i].rank()+1, cube} );
}

int DumpIterator::process(Xtc* xtc, const void* bufEnd)
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

CubeData::CubeData(Detector& det, unsigned nbins) : 
    m_det     (det),
    m_nbins   (nbins),
    m_rawDefV (det.rawDef()),
    m_cubeDefV(0)
{
    size_t bufferSize = nbins*det.cubeBinBytes();
    logging::info("Allocating cube memory for %u bins", nbins);
    logging::info("CubeData  event_buffer_size %lu  cube_buffer_size %lu",
                  m_det.cubeBinBytes(), bufferSize);
    if (bufferSize < (1UL<<32)) {
        //  One datagram
        m_bin_data = new char[bufferSize];
    }
    else {
        //  Multiple datagrams
    }

    unsigned index=det.rawNamesIndex();
    for(auto& v : m_rawDefV) {
        m_rawNames.push_back(NamesId(det.nodeId, index++));
        m_cubeDefV.push_back(CubeDef(v.NameVec));
    }

}

CubeData::~CubeData()
{
    delete[] m_bin_data;
}

//  Initialize the cube from the first event
void CubeData::initialize(Src&       src,
                          const SDV& rawDataV,
                          int        threadNum)
{
    const char* bufEnd = m_bin_data+m_nbins*m_det.cubeBinBytes();
    Dgram& dg = *new (m_bin_data) Dgram( Transition(), Xtc(TypeId(TypeId::Parent,0),src) );
    Xtc& xtc = dg.xtc;
                
    if (threadNum==0) {
        printf("data_init before\n");
        DumpIterator dump((char*)&xtc);
        dump.iterate(&xtc, bufEnd); 
    }
                
    for(unsigned idef=0; idef<m_rawDefV.size(); idef++) {

        //  I can't really skip initialization here
        DescData* rawdata = rawDataV[idef] ?
            new DescData(*rawDataV[idef], 
                         m_det.namesLookup()[m_rawNames[idef]] ) : 0;

        VarDef& rawDef  = m_rawDefV[idef];
        VarDef& cubeDef = m_cubeDefV[idef];

        NamesId namesId(m_det.nodeId, m_det.cubeNamesIndex()+idef);
        //  DescribedData creates the Data container first, then the Shapes container
        DescribedData data(xtc, bufEnd, m_det.namesLookup(), namesId);
        //  Fill the data payload first
        //  bin indices and entries (same for all detectors)
        unsigned size=0;
        {
            uint32_t* p = (uint32_t*)data.data();
            for(unsigned i=0; i<m_nbins; i++)  // bin indices
                p[i] = i;
            memset(&p[m_nbins], 0, m_nbins*sizeof(uint32_t));  // bin entries
            size += 2*m_nbins*sizeof(uint32_t);
        }

        //  Detector-specific payload
        //  Everything here becomes double
        for(unsigned i=0; i<rawDef.NameVec.size(); i++) {
            unsigned rank = rawDef.NameVec[i].rank();
            double_t* dst = (double_t*)((char*)data.data()+size);
            if (rank==0) {
                memset(dst, 0, m_nbins*sizeof(double_t));
                size += m_nbins*sizeof(double_t);
            }
            else {
                if (rawdata==0) {
                    logging::error("Initializing cube vector without raw data");
                    abort();
                }
                uint32_t newShape[XtcData::MaxRank];
                newShape[0] = m_nbins;
                for(unsigned r=0; r<4; r++)
                    newShape[r+1] = rawdata->shape(rawDef.NameVec[i])[r];
                Shape s(newShape);
                unsigned arraySize = s.size(cubeDef.NameVec[i+2]); 
                memset(dst, 0, arraySize);
                size += arraySize;
            }
            // if (threadNum==0)
            //     logging::info("cube: %s dst %p size 0x%lx",rawDef.NameVec[i].name(),dst, size);
        }
        data.set_data_length(size);

        //
        //  Now set the shapes
        //
        uint32_t scalar_array[] = {m_nbins,0,0,0,0};
        Shape scalar(scalar_array);
        data.set_array_shape(CubeDef::bin    , scalar.shape());
        data.set_array_shape(CubeDef::entries, scalar.shape());
        for(unsigned i=2; i<cubeDef.NameVec.size(); i++) {
            unsigned rank = cubeDef.NameVec[i].rank();
            Shape s(scalar_array);
            if (rank>1) {
                uint32_t newShape[XtcData::MaxRank];
                newShape[0] = m_nbins;
                for(unsigned r=0; r<4; r++)
                    newShape[r+1] = rawdata->shape(rawDef.NameVec[i-2])[r];
                s = Shape(newShape);
            }
            data.set_array_shape(i, s.shape());
        }

        //  Printout
        if (threadNum==0) {
            logging::info("Initialized cube with bin %u of %u bins.", bin, m_nbins);
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

    if (threadNum==0) {
        printf("data_init after\n");
        DumpIterator dump((char*)&dg.xtc);
        dump.iterate(&dg.xtc, bufEnd); 
    }
}

//  Add names to the XTC
void CubeData::addNames(unsigned detSegment, Dgram* dgram, const void* bufEnd)
{
    {   printf("Names before\n");
        DumpIterator dump((char*)&dgram->xtc);
        dump.iterate(&dgram->xtc, bufEnd); }

    Alg cubeAlg("cube", 2, 0, 0);
    std::vector<VarDef>& rawDefV = m_rawDefV;
    for(unsigned i=0; i<rawDefV.size(); i++) {
        NamesId rawNamesId (m_det.nodeId,m_det.rawNamesIndex()+i);
        Names& rawNames = m_det.namesLookup()[rawNamesId].names();

        NamesId cubeNamesId(m_det.nodeId,m_det.cubeNamesIndex()+i);
        Names& cubeNames = *new(dgram->xtc, bufEnd)
            Names(bufEnd,
                  rawNames.detName(), cubeAlg,
                  rawNames.detType(), rawNames.detId(), cubeNamesId, detSegment);
        VarDef& cubeDef = m_cubeDefV[i];
        cubeNames.add(dgram->xtc, bufEnd, cubeDef);
        m_det.namesLookup()[cubeNamesId] = NameIndex(cubeNames);
    }

    {   printf("Names after\n");
        DumpIterator dump((char*)&dgram->xtc);
        dump.iterate(&dgram->xtc, bufEnd); }
}

//  Add to the cube
void CubeData::add    (unsigned   bin,
                       const SDV& rawDataV)
{
    Xtc* xtc = (Xtc*)(((Dgram*)m_bin_data)->xtc.payload());

    for(unsigned idef=0; idef<m_rawDefV.size(); idef++) {

        if (rawDataV[idef]) {
#ifdef DBUG
            ShapesData* shpd = rawDataV[idef];
            printf("idef %u  shapesdata %p\n", idef, shpd);
            DUMP_XTC("SHPD", shpd             , (&dgram->xtc));
            DUMP_XTC("DATA", (&shpd->data())  , (&dgram->xtc));
            //DUMP_XTC("SHA",  (&shpd->shapes()), (&dgram->xtc)); // No shapes if no arrays
#endif
            DescData rawdata(*(rawDataV[idef]), 
                             m_det.namesLookup()[m_rawNames[idef]] );

            NamesId cubeNamesId(m_det.nodeId,m_det.cubeNamesIndex()+idef);
            DescData cubedata(*(ShapesData*)xtc, m_det.namesLookup()[cubeNamesId]);
            {
                uint32_t* p = (uint32_t*)cubedata.shapesdata().data().payload();
                p[m_nbins+bin]++;  // increment bin entries
            }
            unsigned size = 2*m_nbins*sizeof(uint32_t);

            VarDef& rawDef = m_rawDefV[idef];
            for(unsigned i=0; i<rawDef.NameVec.size(); i++) {
                Name& name = rawDef.NameVec[i];
                unsigned arraySize = name.rank() ? sizeof(double_t)*Shape(rawdata.shape(name)).num_elements(name.rank()) : sizeof(double_t);
                double* dst = (double_t*)((char*)cubedata.shapesdata().data().payload()+size+bin*arraySize);
                size += m_nbins*arraySize;
                m_det.addToCube(idef, i, 0, dst, rawdata);
            }
        }
                    
        xtc = xtc->next();
    }

}

void CubeData::addSub (unsigned   bin,
                       unsigned   subDet,
                       const SDV& rsd)
{
    Xtc* xtc = (Xtc*)(((Dgram*)m_bin_data)->xtc.payload());

    for(unsigned idef=0; idef<m_rawDefV.size(); idef++) {
        NamesId rawNamesId (m_det.nodeId,m_det.rawNamesIndex ()+idef);
        NamesId cubeNamesId(m_det.nodeId,m_det.cubeNamesIndex()+idef);
        DescData rawdata(*rsd[idef], m_det.namesLookup()[rawNamesId]);  // reading

        DescData cubedata(*(ShapesData*)xtc, m_det.namesLookup()[cubeNamesId]);

        unsigned size = 2*m_nbins*sizeof(uint32_t);

        VarDef& rawDef = m_rawDefV[idef];
        for(unsigned i=0; i<rawDef.NameVec.size(); i++) {
            Name& name = rawDef.NameVec[i];
            unsigned arraySize = name.rank() ? sizeof(double_t)*Shape(rawdata.shape(name)).num_elements(name.rank()) : sizeof(double_t);
            double* dst = (double_t*)((char*)cubedata.shapesdata().data().payload()+size+bin*arraySize);
            size += m_nbins*arraySize;
            m_det.addToCube(idef, i, subDet, dst, rawdata);
        }
        xtc = (Xtc*)(xtc->next());
    }
}

//  Copy one bin into a datagram
void CubeData::copyBin(unsigned      bin,
                       SDV&          shapesDataV,
                       Pds::EbDgram* dg)
{
#ifdef DBUG
    DUMP_DGRAM("TGT DG IN",dg);
#endif

    Xtc* bxtc = (Xtc*)((Dgram*)m_bin_data)->xtc.payload();

    for(unsigned idef=0; idef < m_rawDefV.size(); idef++) {
#ifdef DBUG
        DUMP_XTC("TGT",(&dg->xtc),dg);
        DUMP_XTC("SRC",bxtc,m_bin_data);
#endif
        VarDef& cubeDef = m_cubeDefV[idef];

        NamesId namesId(m_det.nodeId, m_det.cubeNamesIndex()+idef);
        //  DescribedData creates the Data container first, then the Shapes container
        DescribedData data(dg->xtc, dg->xtc.payload()+m_det.cubeBinBytes(), m_det.namesLookup(), namesId);
        shapesDataV.push_back(&data.shapesdata());

        DescData cubedata(*(ShapesData*)bxtc, m_det.namesLookup()[namesId]);
        ((uint32_t*)data.data())[0] = bin;
        ((uint32_t*)data.data())[1] = ((uint32_t*)cubedata.shapesdata().data().payload())[m_nbins+bin]; // entries
        unsigned dstSize = 2*sizeof(uint32_t);
        unsigned srcSize = 2*sizeof(uint32_t)*m_nbins;

        //  The rest are double arrays
        for(unsigned i=2; i<cubeDef.NameVec.size(); i++) {
            Shape s(cubedata.shape(cubeDef.NameVec[i]));
            s.shape()[0] = 1;
            unsigned binSize = s.size(cubeDef.NameVec[i]);
            double_t* dst = (double_t*)((char*)data.data()+dstSize);
            double_t* src = (double_t*)((char*)cubedata.shapesdata().data().payload()+srcSize+bin*binSize);
            memcpy(dst, src, binSize);
            dstSize += binSize;
            srcSize += binSize*m_nbins;
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
}        

//  Add the contents of one bin into a datagram
void CubeData::addBin (unsigned      bin,
                       const SDV&    shapesDataV,
                       Pds::EbDgram* dg)
{
    Xtc& bxtc = ((Dgram*)m_bin_data)->xtc;

    for(unsigned idef=0; idef < m_rawDefV.size(); idef++) {
        VarDef& cubeDef = m_cubeDefV[idef];

        NamesId namesId(m_det.nodeId, m_det.cubeNamesIndex()+idef);
        DescData cubedata(*(ShapesData*)(bxtc.payload()), m_det.namesLookup()[namesId]);
        void* data = shapesDataV[idef]->data().payload();
        ((uint32_t*)data)[1] += ((uint32_t*)(cubedata.shapesdata().data().payload()))[m_nbins+bin]; // entries
        unsigned dstSize = 2*sizeof(uint32_t);
        unsigned srcSize = 2*sizeof(uint32_t)*m_nbins;

        //  The rest are double arrays
        for(unsigned i=2; i<cubeDef.NameVec.size(); i++) {
            Shape s(cubedata.shape(cubeDef.NameVec[i]));
            s.shape()[0] = 1;
            unsigned binSize = s.size(cubeDef.NameVec[i]);
            double_t* dst = (double_t*)((char*)data+dstSize);
            double_t* src = (double_t*)((char*)cubedata.shapesdata().data().payload()+srcSize+bin*binSize);
                
            for(unsigned j=0; j<binSize/sizeof(double_t); j++)
                dst[j] += src[j];

            dstSize += binSize;
            srcSize += binSize*m_nbins;
        }

        bxtc = *(bxtc.next());
    }
}

//  Get the cube as a datagram for EndRun
Dgram* CubeData::dgram(XtcData::Dgram* dgram, TransitionId::Value transitionId)
{
    unsigned nbins = m_nbins;
    { Dgram* bindg = (Dgram*)m_bin_data;
        DumpIterator dump((char*)bindg);
        dump.iterate(&bindg->xtc, m_bin_data+nbins*m_det.cubeBinBytes()); }

    Dgram* dg = (Dgram*)m_bin_data;

#ifdef DBUG
    DUMP_DGRAM("TGT0 DG IN",dg);
    DUMP_XTC("TGT0",(&dg->xtc),dg);
#endif

    //  Overwrite the header
    new(m_bin_data) Transition(dgram->type(), transitionId, dgram->time, dgram->env);

#ifdef DBUG
    DUMP_DGRAM("TGT DG IN",dg);
#endif

    return dg;
}

//  Add this cube to the data in the datagram
void CubeData::add(XtcData::Dgram* dg)
{
    Xtc* xtc = (Xtc*)dg->xtc.payload();
    Xtc* bxtc = (Xtc*)((Dgram*)m_bin_data)->xtc.payload();

    for(unsigned idef=0; idef < m_rawDefV.size(); idef++) {
#ifdef DBUG
        DUMP_XTC("TGT",xtc,dg);
        DUMP_XTC("SRC",bxtc,m_bin_data);
#endif
        VarDef& cubeDef = m_cubeDefV[idef];
                        
        NamesId namesId(m_det.nodeId, m_det.cubeNamesIndex()+idef);
        //  Sum the data
        void* data = ((ShapesData*)xtc)->data().payload();
        DescData cubedata(*(ShapesData*)bxtc, m_det.namesLookup()[namesId]);

        // entries
        for(unsigned bin=0; bin<m_nbins; bin++)
            ((uint32_t*)data)[m_nbins+bin] += ((uint32_t*)(cubedata.shapesdata().data().payload()))[m_nbins+bin];
        unsigned dstSize = 2*sizeof(uint32_t)*m_nbins;
        unsigned srcSize = 2*sizeof(uint32_t)*m_nbins;

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
}

std::vector<Dgram*> CubeData::dgram(Dgram&    dg,
                                    unsigned& binsPerDg)
{
    return std::vector<Dgram*>(0);
}

void CubeData::add(std::vector<Dgram*>& dg,
                   unsigned             binsPerDg)
{
}
