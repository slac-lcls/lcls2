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
} _fexDef;

class RawDef : public VarDef
{
public:
    enum index
    {
        value,
        array_raw
    };

    RawDef()
    {
        Alg raw("raw", 1, 0, 0);
        NameVec.push_back({"value",     Name::UINT32, 0, raw});
        NameVec.push_back({"array_raw", Name::UINT16, 2, raw});
    }
} _rawDef;

AreaDetector::AreaDetector(Parameters* para, MemPool* pool) :
    XpmDetector(para, pool),
    m_constants(NumConstants,(char*)0)
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
    fexNames.add(xtc, bufEnd, _fexDef);
    m_namesLookup[fexNamesId] = NameIndex(fexNames);

    Alg rawAlg("raw", 2, 0, 0);
    NamesId rawNamesId(nodeId,RawNamesIndex);
    Names& rawNames = *new(xtc, bufEnd) Names(bufEnd,
                                              m_para->detName.c_str(), rawAlg,
                                              m_para->detType.c_str(), m_para->serNo.c_str(), rawNamesId, m_para->detSegment);
    rawNames.add(xtc, bufEnd, _rawDef);
    m_namesLookup[rawNamesId] = NameIndex(rawNames);

    return 0;
}

unsigned AreaDetector::beginrun(XtcData::Xtc& xtc, const void* bufEnd, const json& runInfo)
{
    logging::info("AreaDetector beginrun");

    //  Fetch new constants for cube (via Python)
    //  sim_length is in units of 32-bit words, 
    //  but image is interpreted as 16-bit words (see RawDef)
    std::map<std::string,std::string>::iterator it = m_para->kwargs.find("sw_sim_length");
    unsigned constants_length = sizeof(uint32_t)/sizeof(uint16_t) * std::popcount(m_para->laneMask)
        * (it == m_para->kwargs.end() ? m_length : std::stoul(it->second.data()));

    if (m_constants[Pedestals])
        delete m_constants[Pedestals];
    m_constants[Pedestals] = new char[sizeof(double_t)*constants_length];

    if (m_constants[Gains])
        delete m_constants[Gains];
    m_constants[Gains] = new char[sizeof(double_t)*constants_length];

    //  Give some interesting shape
    for(unsigned i=0; i<constants_length; i++) {
        ((double_t*)m_constants[Pedestals])[i] = 0.5;
        ((double_t*)m_constants[Gains])    [i] = 1.;
        //        ((double_t*)m_constants[Pedestals])[i] = 0.5*double(constants_length-i);
        //        ((double_t*)m_constants[Gains])    [i] = 100.-double(10*i)/double(constants_length);
    }

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
    std::map<std::string,std::string>::iterator it = m_para->kwargs.find("sw_sim_length");
    NamesId rawNamesId(nodeId,RawNamesIndex);
    DescribedData raw(dgram.xtc, bufEnd, m_namesLookup, rawNamesId);
    unsigned size = 0;
    unsigned nlanes = 0;
    *(uint32_t*)raw.data() = 9;
    size += sizeof(uint32_t);
    for (int i=0; i<PGP_MAX_LANES; i++) {
        if (event->mask & (1 << i)) {
            // size without Event header
            // add sw_sim_length kwarg to simulate larger detectors even if the pcie bandwidth
            //   doesn't support it
            int dataSize = (it==m_para->kwargs.end()) ? 
                event->buffers[i].size - 32 : std::stoul(it->second.data())*sizeof(uint32_t);
            uint32_t dmaIndex = event->buffers[i].index;
            uint8_t* rawdata = ((uint8_t*)m_pool->dmaBuffers[dmaIndex]) + 32;

            memcpy((uint8_t*)raw.data() + size, (uint8_t*)rawdata, dataSize);
            size += dataSize;
            nlanes++;
         }
     }
    raw.set_data_length(size);

    ((uint32_t*)raw.data())[RawDef::value] = 9;
    size = (size - sizeof(uint32_t)) / (nlanes * sizeof(uint16_t));
    unsigned raw_shape[MaxRank] = {nlanes, size};
    raw.set_array_shape(RawDef::array_raw, raw_shape);
}

VarDef   AreaDetector::rawDef () 
{
    return _rawDef;
}

void AreaDetector::addToCube(unsigned rawDefIndex, unsigned subIndex, double_t* dst, DescData& rawData)
{
    switch(rawDefIndex) {
    case RawDef::value:
        if (subIndex==0)
            *dst += double(rawData.get_value<uint32_t>(RawDef::value));
        break;
    case RawDef::array_raw:
        {
            Array<uint16_t> rawArrT = rawData.get_array<uint16_t>(rawDefIndex);
            Array<double_t> calArrT((char*)dst, rawArrT.shape(), rawArrT.rank());
            Array<double_t> pedArrT (m_constants[Pedestals], rawArrT.shape(), rawArrT.rank());
            Array<double_t> gainArrT(m_constants[Gains], rawArrT.shape(), rawArrT.rank());

            //  Apply the calibration, so shape matters
            //            unsigned ix=subIndex;
            for(unsigned ix=0; ix<rawArrT.shape()[0]; ix++)
            for(unsigned iy=0; iy<rawArrT.shape()[1]; iy++)
                calArrT(ix,iy) += gainArrT(ix,iy)*(double( rawArrT(ix,iy) ) - pedArrT(ix,iy));

            break;
        }
    default:
        // error
        printf("*** %s:%d: unknown RawDef array %u\n",__FILE__,__LINE__,rawDefIndex);
        abort();
    }
}
}
