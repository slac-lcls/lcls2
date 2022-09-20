#include "AreaDetector.hh"
#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
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

    return 0;
}

unsigned AreaDetector::beginrun(XtcData::Xtc& xtc, const void* bufEnd, const json& runInfo)
{
    logging::info("AreaDetector beginrun");
    return 0;
}

void AreaDetector::event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event)
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

}
