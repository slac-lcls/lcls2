#include "AreaDetector.hh"
#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include "psalg/utils/SysLog.hh"

using namespace XtcData;
using json = nlohmann::json;
using logging = psalg::SysLog;

namespace Drp {

class FexDef : public VarDef
{
public:
    enum index
    {
        array_fex
    };

    FexDef()
    {
        Alg fex("fex", 1, 0, 0);
        NameVec.push_back({"array_fex", Name::UINT16, 2, fex});
    }
};

class RawDef : public VarDef
{
public:
    enum index
    {
        array_raw
    };

    RawDef()
    {
        Alg raw("raw", 1, 0, 0);
        NameVec.push_back({"array_raw", Name::UINT16, 2, raw});
    }
};

class CubeDef : public VarDef
{
public:
    enum index
    {
        array
    };

    CubeDef()
    {
        Alg cube("cube", 1, 0, 0);
        NameVec.push_back({"array"  , Name::DOUBLE, 2, cube});
    }
};

class MyIterator : public XtcIterator {
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

AreaDetector::AreaDetector(Parameters* para, MemPool* pool) :
    XpmDetector(para, pool)
{
}

unsigned AreaDetector::configure(const std::string& config_alias, Xtc& xtc, const void* bufEnd)
{
    logging::info("AreaDetector configure");

    if (XpmDetector::configure(config_alias, xtc, bufEnd))
        return 1;

    Alg fexAlg("fex", 2, 0, 0);
    NamesId fexNamesId(nodeId,FexNamesIndex);
    Names& fexNames = *new(xtc, bufEnd) Names(bufEnd,
                                              m_para->detName.c_str(), fexAlg,
                                              m_para->detType.c_str(), m_para->serNo.c_str(), fexNamesId, m_para->detSegment);
    FexDef myFexDef;
    fexNames.add(xtc, bufEnd, myFexDef);
    m_namesLookup[fexNamesId] = NameIndex(fexNames);

    Alg rawAlg("raw", 2, 0, 0);
    NamesId rawNamesId(nodeId,RawNamesIndex);
    Names& rawNames = *new(xtc, bufEnd) Names(bufEnd,
                                              m_para->detName.c_str(), rawAlg,
                                              m_para->detType.c_str(), m_para->serNo.c_str(), rawNamesId, m_para->detSegment);
    RawDef myRawDef;
    rawNames.add(xtc, bufEnd, myRawDef);
    m_namesLookup[rawNamesId] = NameIndex(rawNames);

    Alg cubeAlg("cube", 2, 0, 0);
    NamesId cubeNamesId(nodeId,CubeNamesIndex);
    Names& cubeNames = *new(xtc, bufEnd) Names(bufEnd,
                                              m_para->detName.c_str(), cubeAlg,
                                              m_para->detType.c_str(), m_para->serNo.c_str(), cubeNamesId, m_para->detSegment);
    CubeDef myCubeDef;
    cubeNames.add(xtc, bufEnd, myCubeDef);
    m_namesLookup[cubeNamesId] = NameIndex(cubeNames);

    return 0;
}

unsigned AreaDetector::beginrun(XtcData::Xtc& xtc, const void* bufEnd, const json& runInfo)
{
    logging::info("AreaDetector beginrun");
    return 0;
}

void AreaDetector::event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t l1count)
{
    // fex data
    NamesId fexNamesId(nodeId,FexNamesIndex);
    CreateData fex(dgram.xtc, bufEnd, m_namesLookup, fexNamesId);
    unsigned shape[MaxRank] = {3,3};
    Array<uint16_t> arrayT = fex.allocate<uint16_t>(FexDef::array_fex,shape);

    // int index = __builtin_ffs(pgp_data->buffer_mask) - 1;
    // Pds::TimingHeader* timing_header = reinterpret_cast<Pds::TimingHeader*>(pgp_data->buffers[index].data);
    // uint32_t* rawdata = (uint32_t*)(timing_header+1);

    /*
    int nelements = (buffers[l].size - 32) / 4;
    int64_t sum = 0L;
    for (int i=1; i<nelements; i++) {
        sum += rawdata[i];
    }

    int64_t result = nelements rawdata[1] * (nelements -1) / 2;
    if (sum != result) {
        printf("Error in worker calculating sum of the image\n");
        printf("%l %l\n", sum, result);
    }
    printf("raw data %u %u %u %u %u\n", rawdata[0], rawdata[1], rawdata[2], rawdata[3], rawdata[4]);
    */

    for(unsigned i=0; i<shape[0]; i++){
        for (unsigned j=0; j<shape[1]; j++) {
            arrayT(i,j) = i+j;
        }
    }

    // raw data
    NamesId rawNamesId(nodeId,RawNamesIndex);
    DescribedData raw(dgram.xtc, bufEnd, m_namesLookup, rawNamesId);
    unsigned size = 0;
    unsigned nlanes = 0;
    for (int i=0; i<PGP_MAX_LANES; i++) {
        if (event->mask & (1 << i)) {
            // size without Event header
            int dataSize = event->buffers[i].size - 32;
            uint32_t dmaIndex = event->buffers[i].index;
            uint8_t* rawdata = ((uint8_t*)m_pool->dmaBuffers[dmaIndex]) + 32;

            memcpy((uint8_t*)raw.data() + size, (uint8_t*)rawdata, dataSize);
            size += dataSize;
            nlanes++;
         }
     }
    raw.set_data_length(size);
    unsigned raw_shape[MaxRank] = {nlanes, size / nlanes / 2};
    raw.set_array_shape(RawDef::array_raw, raw_shape);
}

static void _dumpNames(DescData& data, const char* title)
{
    Names& names = data.nameindex().names();
    logging::debug("Found %d names in %s",names.num(), title);
    for(unsigned i=0; i<names.num(); i++) {
        Name& name = names.get(i);
        logging::debug("\t%d : %s",i,name.name());
    }
}

void AreaDetector::cube(XtcData::Dgram& dgram, unsigned binId, void* binXtc, unsigned& entries, const void* bufEnd)
{
    logging::debug("AreaDetector::cube dgram (%p) bin (%u) bin_xtc (%p) bin_entries (%u) bufEnd (%p)",
                   &dgram, binId, binXtc, entries, bufEnd);

    //  Navigate to the raw data
    //    Can either be direct (knowing the _event method)
    //    or use an iterator
    NamesId rawNames = NamesId(nodeId,RawNamesIndex);
    Name name = RawDef().NameVec[RawDef::array_raw];

    MyIterator iter(rawNames);
    iter.iterate(&dgram.xtc, dgram.xtc.next());
    if (!iter.shapesdata()) {
        logging::error("Unable to find raw data in xtc");
        return;
    }

    {
        ShapesData& sh = *iter.shapesdata();
        logging::debug("AreaDetector::cube iter.shapesdata: src %04x ctns %04x ext %08x",
                       sh.src, sh.contains.value(), sh.extent);
        Xtc& first = *(Xtc*)sh.payload();
        logging::debug("AreaDetector::cube iter.shapesdata.first: src %04x ctns %04x ext %08x",                   
                       first.src, first.contains.value(), first.extent);
        Xtc& secnd = *first.next();
        logging::debug("AreaDetector::cube iter.shapesdata.secnd: src %04x ctns %04x ext %08x",                   
                       secnd.src, secnd.contains.value(), secnd.extent);
    }

    Shape& shape = iter.shapesdata()->shapes().get(RawDef::array_raw);
    unsigned nelems = shape.size(name)/name.get_element_size(name.type());
    DescData rawdata(*(iter.shapesdata()), m_namesLookup[rawNames]);
    Array<uint8_t> rawArrT = rawdata.get_array<uint8_t>(RawDef::array_raw);

    logging::debug("shape [%u, %u, %u, %u, %u]  nelems (%u)", 
                   shape.shape()[0],
                   shape.shape()[1],
                   shape.shape()[2],
                   shape.shape()[3],
                   shape.shape()[4], nelems);
    logging::debug("rawArrT  rank (%u) data (%p) nelems (%u)",
                   rawArrT.rank(),
                   rawArrT.data(),
                   rawArrT.num_elem());
    
    // some simple checks
    _dumpNames(rawdata,"raw");

    NamesId cubeNamesId(nodeId,CubeNamesIndex);
    if (!entries) {
        Xtc& xtc = *new ((char*)binXtc, bufEnd) Xtc(TypeId(TypeId::Parent,0),dgram.xtc.src);
        //  Create the shapes data within the provided xtc
        CreateData data(xtc, bufEnd, m_namesLookup, cubeNamesId);
        _dumpNames(data,"cube");
        Array<double_t> arrayT = data.allocate<double_t>(CubeDef::array, shape.shape());
        logging::debug("binArrT  rank (%u) data (%p) nelems (%u)",
                       arrayT.rank(),
                       arrayT.data(),
                       arrayT.num_elem());
        for(unsigned i=0; i<shape.shape()[0]; i++)
            for(unsigned j=0; j<shape.shape()[1]; j++)
                arrayT(i,j) = double(rawArrT(i,j));
    }
    else {
        //  Extract the shapes data from the provided xtc
        //  This only works if the array is the only member in CubeDef;
        //    i.e. the _offsets vector is not reconstituted.
        Xtc& xtc = *(Xtc*)(binXtc);
        DescData data(*(ShapesData*)xtc.payload(), m_namesLookup[cubeNamesId]);
        _dumpNames(data,"cube");
        Array<double_t> arrayT = data.get_array<double_t>(CubeDef::array);
        logging::debug("binArrT  rank (%u) data (%p) nelems (%u)",
                       arrayT.rank(),
                       arrayT.data(),
                       arrayT.num_elem());
        //  calibrate and sum
        for(unsigned i=0; i<shape.shape()[0]; i++)
            for(unsigned j=0; j<shape.shape()[1]; j++)
                arrayT(i,j) += double(rawArrT(i,j));
    }
}

}
