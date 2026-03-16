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

CubeData::CubeData(Detector& det, unsigned nbins, unsigned poolBufferSize) : 
    m_det     (det),
    m_nbins   (nbins),
    m_rawDefV (det.rawDef()),
    m_cubeDefV(0)
{
    unsigned index = det.rawNamesIndex();
    unsigned nshapes = 0;
    for(auto& v : m_rawDefV) {
        m_rawNames.push_back(NamesId(det.nodeId, index++));
        m_cubeDefV.push_back(CubeDef(v.NameVec));
        nshapes += v.NameVec.size();
    }

    //  Want to prevent xtc.extent exceeding 32 bits
    //  Parent XTC + ShapesData/Shapes/Data for each rawDef
    size_t overhead    = sizeof(Dgram) + m_rawDefV.size()*3*sizeof(Xtc) + nshapes*sizeof(Shape);
    size_t maxDataSize = (1UL<<31) - overhead;
    size_t dataSize    = det.cubeBinBytes();
    dataSize *= nbins;

    unsigned nBuffers = (dataSize / maxDataSize) + 1;
    m_binsPerBuf = (nbins+nBuffers-1)/nBuffers;
    m_bufferSize = overhead+m_binsPerBuf*(det.cubeBinBytes()+2*sizeof(uint32_t));
    for(unsigned i=0; i<nBuffers; i++)
        m_buffer.push_back( new char[m_bufferSize] );

    logging::info("Allocating cube memory for %u bins [%lu]", nbins, dataSize);
    logging::info("CubeData  cube_dgrams %u  cube_buffer_size %lu",
                  nBuffers, m_bufferSize);

    unsigned bufferSize = overhead + det.cubeBinBytes();
    if (nshapes == 0)
        bufferSize = sizeof(Dgram) + 3*sizeof(Xtc) + 2*sizeof(Shape) + nbins*2*sizeof(uint32_t);
    bufferSize += poolBufferSize;

    m_bufferBin = new char[bufferSize];
    m_bufferBinSize = bufferSize;

    logging::info("CubeData  cube_bin_buffer_size %lu",
                  m_bufferBinSize);
}

CubeData::~CubeData()
{
    for(unsigned i=0; i<m_buffer.size(); i++)
        delete[] m_buffer[i];

    delete[] m_bufferBin;
}

//  Initialize the cube from the first event
void CubeData::initialize(Src&       src,
                          const SDV& rawDataV,
                          int        threadNum)
{
    for(unsigned ibuff=0; ibuff<m_buffer.size(); ibuff++) {
        //  The bin range covered by datagram[ibuff]
        unsigned ibin  = ibuff * m_binsPerBuf;
        unsigned nbins = m_binsPerBuf;
        if (ibuff == m_buffer.size()-1) {
            nbins = m_nbins - ibin;
        }

        const char* bufEnd = m_buffer[ibuff]+m_bufferSize;
        Dgram& dg = *new (m_buffer[ibuff]) Dgram( Transition(), Xtc(TypeId(TypeId::Parent,0),src) );
        Xtc& xtc = dg.xtc;
                
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
                for(unsigned i=0; i<nbins; i++)  // bin indices
                    p[i] = ibin+i;
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
                // if (threadNum==0)
                //     logging::info("cube: %s dst %p size 0x%lx",rawDef.NameVec[i].name(),dst, size);
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
                logging::info("Initialized cube[%u] with %u bins.", ibuff, nbins);
                ShapesData& shd = data.shapesdata();
                logging::info("Shapes at %lx,  Data at %lx",
                              (char*)&shd.shapes()-xtc.payload(), 
                              (char*)&shd.data  ()-xtc.payload());
                if (ibuff==0) {
                    for(unsigned i=0; i<cubeDef.NameVec.size(); i++) {
                        uint32_t* sh = shd.shapes().get(i).shape();
                        logging::info("Shape[%u] (%s): %u %u %u %u %u",
                                      i, cubeDef.NameVec[i].name(), sh[0], sh[1], sh[2], sh[3], sh[4]);
                    } 
                }
            }
        } // for(idef...

        if (threadNum==0 && ibuff==0) {
            printf("data_init after\n");
            DumpIterator dump((char*)&dg.xtc);
            dump.iterate(&dg.xtc, bufEnd); 
        }

        ibin += nbins;
    } // ibuff
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
    unsigned nbins = m_binsPerBuf;
    unsigned ibuff = bin/nbins;
    unsigned ibin  = bin%nbins;
    Xtc* xtc = (Xtc*)(((Dgram*)m_buffer[ibuff])->xtc.payload());

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
                p[nbins+ibin]++;  // increment bin entries
            }
            unsigned size = 2*nbins*sizeof(uint32_t);

            VarDef& rawDef = m_rawDefV[idef];
            for(unsigned i=0; i<rawDef.NameVec.size(); i++) {
                Name& name = rawDef.NameVec[i];
                unsigned arraySize = name.rank() ? sizeof(double_t)*Shape(rawdata.shape(name)).num_elements(name.rank()) : sizeof(double_t);
                double* dst = (double_t*)((char*)cubedata.shapesdata().data().payload()+size+ibin*arraySize);
                size += nbins*arraySize;
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
    unsigned nbins = m_binsPerBuf;
    unsigned ibuff = bin/nbins;
    unsigned ibin  = bin%nbins;
    Xtc* xtc = (Xtc*)(((Dgram*)m_buffer[ibuff])->xtc.payload());

    for(unsigned idef=0; idef<m_rawDefV.size(); idef++) {
        NamesId rawNamesId (m_det.nodeId,m_det.rawNamesIndex ()+idef);
        NamesId cubeNamesId(m_det.nodeId,m_det.cubeNamesIndex()+idef);
        DescData rawdata(*rsd[idef], m_det.namesLookup()[rawNamesId]);  // reading

        DescData cubedata(*(ShapesData*)xtc, m_det.namesLookup()[cubeNamesId]);

        unsigned size = 2*nbins*sizeof(uint32_t);

        VarDef& rawDef = m_rawDefV[idef];
        for(unsigned i=0; i<rawDef.NameVec.size(); i++) {
            Name& name = rawDef.NameVec[i];
            unsigned arraySize = name.rank() ? sizeof(double_t)*Shape(rawdata.shape(name)).num_elements(name.rank()) : sizeof(double_t);
            double* dst = (double_t*)((char*)cubedata.shapesdata().data().payload()+size+ibin*arraySize);
            size += nbins*arraySize;
            m_det.addToCube(idef, i, subDet, dst, rawdata);
        }
        xtc = (Xtc*)(xtc->next());
    }
}

//  Copy one bin into a datagram (unless we are the timing system)
Pds::EbDgram* CubeData::copyBin(unsigned      bin,
                                SDV&          shapesDataV,
                                Pds::EbDgram* dg)
{
    //  Append the intermediate bin sum to the pebble data
    memcpy(m_bufferBin, dg, sizeof(*dg)+dg->xtc.sizeofPayload());
    dg = (EbDgram*)m_bufferBin;

#ifdef DBUG
    DUMP_DGRAM("TGT DG IN",dg);
#endif

    unsigned nbins = m_binsPerBuf;
    unsigned ibuff = bin/nbins;
    unsigned ibin  = bin%nbins;
    Xtc* bxtc = (Xtc*)(((Dgram*)m_buffer[ibuff])->xtc.payload());

    //  Check for timing system
    if (m_rawDefV[0].NameVec.size()==0) {
        NamesId namesId(m_det.nodeId, m_det.cubeNamesIndex());
        //  DescribedData creates the Data container first, then the Shapes container
        unsigned sz = 2*sizeof(uint32_t)*nbins+2*sizeof(Shape)+3*sizeof(Xtc);
        DescribedData data(dg->xtc, (char*)dg->xtc.next()+sz, m_det.namesLookup(), namesId);
        shapesDataV.push_back(&data.shapesdata());

        DescData cubedata(*(ShapesData*)bxtc, m_det.namesLookup()[namesId]);
        memcpy(data.data(), cubedata.shapesdata().data().payload(), 2*nbins*sizeof(uint32_t));
        unsigned dstSize = 2*sizeof(uint32_t)*nbins;
        data.set_data_length(dstSize);

        //  Now set the shapes
        uint32_t scalar_array[] = {nbins,0,0,0,0};
        Shape    scalar(scalar_array);
        data.set_array_shape(CubeDef::bin    , scalar_array);
        data.set_array_shape(CubeDef::entries, scalar_array);
    }
    else {
        //  Not timing system
        for(unsigned idef=0; idef < m_rawDefV.size(); idef++) {
#ifdef DBUG
            DUMP_XTC("TGT",(&dg->xtc),dg);
            DUMP_XTC("SRC",bxtc,m_buffer[ibuff]);
#endif
            VarDef& cubeDef = m_cubeDefV[idef];

            NamesId namesId(m_det.nodeId, m_det.cubeNamesIndex()+idef);
            //  DescribedData creates the Data container first, then the Shapes container
            DescribedData data(dg->xtc, dg->xtc.payload()+m_det.cubeBinBytes(), m_det.namesLookup(), namesId);
            shapesDataV.push_back(&data.shapesdata());

            DescData cubedata(*(ShapesData*)bxtc, m_det.namesLookup()[namesId]);
            ((uint32_t*)data.data())[0] = bin;
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
    }
#ifdef DBUG
    DUMP_DGRAM("TGT DG OUT",dg);
    DUMP_XTC("TGT OUT",(&dg->xtc),dg);
#endif
    return dg;
}        

//  Add the contents of one bin into a datagram (unless we are the timing system)
void CubeData::addBin (unsigned      bin,
                       const SDV&    shapesDataV,
                       Pds::EbDgram* dg)
{
    unsigned nbins = m_binsPerBuf;
    unsigned ibuff = bin/nbins;
    unsigned ibin  = bin%nbins;
    Xtc& bxtc = ((Dgram*)m_buffer[ibuff])->xtc;

    //  Check for timing system
    if (m_rawDefV[0].NameVec.size()==0) {
        NamesId namesId(m_det.nodeId, m_det.cubeNamesIndex());
        DescData cubedata(*(ShapesData*)(bxtc.payload()), m_det.namesLookup()[namesId]);
        void* data = shapesDataV[0]->data().payload();
        for(unsigned i=0; i<nbins; i++)
            ((uint32_t*)data)[nbins+i] += ((uint32_t*)(cubedata.shapesdata().data().payload()))[nbins+i]; // entries
    }
    else {
        //  Not timing system
        for(unsigned idef=0; idef < m_rawDefV.size(); idef++) {
            VarDef& cubeDef = m_cubeDefV[idef];

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
    }
}

//  Get the cube as a datagram for EndRun
std::vector<Dgram*> CubeData::dgram(XtcData::Dgram* dgram, TransitionId::Value transitionId)
{
    std::vector<Dgram*> result;
    for(unsigned ibuff=0; ibuff<m_buffer.size(); ibuff++) {
        //  Overwrite the header
        unsigned env = dgram->env | (ibuff==(m_buffer.size()-1) ? 0 : (1<<16));
        new(m_buffer[ibuff]) Transition(dgram->type(), transitionId, dgram->time, env); // dgram->env);
        result.push_back((Dgram*)m_buffer[ibuff]);
    }

    return result;
}

//  Add this cube to the data in the datagram
void CubeData::add(std::vector<XtcData::Dgram*>& dg)
{
    for(unsigned ibuff=0; ibuff<m_buffer.size(); ibuff++) {
        //  The bin range covered by datagram[ibuff]
        unsigned ibin  = ibuff * m_binsPerBuf;
        unsigned nbins = m_binsPerBuf;
        if (ibuff == m_buffer.size()-1) {
            nbins = m_nbins - ibin;
        }

        Xtc* xtc  = (Xtc*)dg[ibuff]->xtc.payload();
        Xtc* bxtc = (Xtc*)((Dgram*)m_buffer[ibuff])->xtc.payload();

        for(unsigned idef=0; idef < m_rawDefV.size(); idef++) {
#ifdef DBUG
            DUMP_XTC("TGT",xtc,dg[ibuff]);
            DUMP_XTC("SRC",bxtc,m_buffer[ibuff]);
#endif
            VarDef& cubeDef = m_cubeDefV[idef];
                        
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
        DUMP_DGRAM("TGT DG OUT",dg[ibuff]);
        DUMP_XTC("TGT OUT",(&dg[ibuff]->xtc),dg[ibuff]);
#endif
    }
}

