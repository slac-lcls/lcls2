
#include "AreaDetector.hh"
#include "xtcdata/xtc/VarDef.hh"

using namespace XtcData;

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

void AreaDetector::configure(Xtc& parent)
{
    Alg cspadFexAlg("cspadFexAlg", 1, 2, 3);
    unsigned segment = 0;
    Names& fexNames = *new(parent) Names("xppcspad", cspadFexAlg, "cspad", "detnum1234", segment);
    fexNames.add(parent, myFexDef);
    m_namesVec.push_back(NameIndex(fexNames));

    Alg cspadRawAlg("cspadRawAlg", 1, 2, 3);
    Names& rawNames = *new(parent) Names("xppcspad", cspadRawAlg, "cspad", "detnum1234", segment);
    rawNames.add(parent, myRawDef);
    m_namesVec.push_back(NameIndex(rawNames));
}

AreaDetector::AreaDetector() : m_evtcount(0) {}

void AreaDetector::event(Xtc& parent, PGPData* pgp_data)
{
    m_evtcount+=1;
    int index = __builtin_ffs(pgp_data->buffer_mask) - 1;
    Transition* event_header = reinterpret_cast<Transition*>(pgp_data->buffers[index]->virt);

    unsigned nameId=0;
    CreateData fex(parent, m_namesVec, nameId);
    unsigned shape[Name::MaxRank] = {3,3};
    Array<uint16_t> arrayT = fex.allocate<uint16_t>(FexDef::array_fex,shape);
    uint16_t* rawdata = (uint16_t*)(event_header+1);
    for(unsigned i=0; i<shape[0]; i++){
        for (unsigned j=0; j<shape[1]; j++) {
            arrayT(i,j) = i+j;
        }
    }

    if (m_evtcount%10==0) {
        nameId=1;
        CreateData raw(parent, m_namesVec, nameId);
        unsigned shape[Name::MaxRank] = {5,5};
        Array<uint16_t> arrayT = raw.allocate<uint16_t>(RawDef::array_raw,shape);
        for(unsigned i=0; i<shape[0]; i++){
            for (unsigned j=0; j<shape[1]; j++) {
                arrayT(i,j) = i+j;
            }
        }
    }
}
