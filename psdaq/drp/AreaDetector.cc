
#include "AreaDetector.hh"
#include "TimingHeader.hh"
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
static FexDef myFexDef;

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
static RawDef myRawDef;

class RunInfoDef : public VarDef
{
public:
  enum index
    {
        EXPT,
        RUNNUM
    };

  RunInfoDef()
   {
       NameVec.push_back({"expt",Name::CHARSTR,1});
       NameVec.push_back({"runnum",Name::UINT32});
   }
};

static RunInfoDef myRunInfoDef;

AreaDetector::AreaDetector(Parameters* para, MemPool* pool) :
    XpmDetector(para, pool), m_evtcount(0)
{
}

unsigned AreaDetector::configure(const std::string& config_alias, Xtc& xtc)
{
    logging::info("AreaDetector configure");

    Alg cspadFexAlg("cspadFexAlg", 1, 2, 3);
    unsigned segment = 0;
    NamesId fexNamesId(nodeId,FexNamesIndex);
    Names& fexNames = *new(xtc) Names("xppcspad", cspadFexAlg, "cspad",
                                            "detnum1234", fexNamesId, segment);
    fexNames.add(xtc, myFexDef);
    m_namesLookup[fexNamesId] = NameIndex(fexNames);

    Alg cspadRawAlg("cspadRawAlg", 1, 2, 3);
    NamesId rawNamesId(nodeId,RawNamesIndex);
    Names& rawNames = *new(xtc) Names("xppcspad", cspadRawAlg, "cspad",
                                            "detnum1234", rawNamesId, segment);
    rawNames.add(xtc, myRawDef);
    m_namesLookup[rawNamesId] = NameIndex(rawNames);

    // beginrun support
    Alg runInfoAlg("runinfo",0,0,1);
    NamesId runInfoNamesId(nodeId, RunInfoNamesIndex);
    Names& runInfoNames = *new(xtc) Names("runinfo", runInfoAlg, "runinfo", "", runInfoNamesId, segment);
    runInfoNames.add(xtc, myRunInfoDef);
    m_namesLookup[runInfoNamesId] = NameIndex(runInfoNames);

    return 0;
}

unsigned AreaDetector::beginrun(XtcData::Xtc& xtc, const json& runInfo)
{
    logging::info("AreaDetector beginrun");

    std::string experiment_name;
    unsigned int run_number = 0;
    if (runInfo.find("run_info") != runInfo.end()) {
        if (runInfo["run_info"].find("experiment_name") != runInfo["run_info"].end()) {
            experiment_name = runInfo["run_info"]["experiment_name"];
        }
        if (runInfo["run_info"].find("run_number") != runInfo["run_info"].end()) {
            run_number = runInfo["run_info"]["run_number"];
        }
    }
    logging::debug("%s: expt=\"%s\" runnum=%u",
                   __PRETTY_FUNCTION__, experiment_name.c_str(), run_number);

    if (run_number > 0) {
        // runinfo data
        NamesId runInfoNamesId(nodeId, RunInfoNamesIndex);
        CreateData runinfo(xtc, m_namesLookup, runInfoNamesId);
        runinfo.set_string(RunInfoDef::EXPT, experiment_name.c_str());
        runinfo.set_value(RunInfoDef::RUNNUM, (uint32_t)run_number);
    }
    return 0;
}

void AreaDetector::event(XtcData::Dgram& dgram, PGPEvent* event)
{
    m_evtcount+=1;

    // fex data
    NamesId fexNamesId(nodeId,FexNamesIndex);
    CreateData fex(dgram.xtc, m_namesLookup, fexNamesId);
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
    DescribedData raw(dgram.xtc, m_namesLookup, rawNamesId);
    unsigned size = 0;
    unsigned nlanes = 0;
    for (int i=0; i<4; i++) {
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
