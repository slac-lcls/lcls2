
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

void AreaDetector::configure(Dgram& dgram, PGPData* pgp_data)
{
    printf("AreaDetector configure\n");

    // copy Event header into beginning of Datagram
    int index = __builtin_ffs(pgp_data->buffer_mask) - 1;
    printf("index %d\n", index);
    Transition* event_header = reinterpret_cast<Transition*>(pgp_data->buffers[index].data);
    memcpy(&dgram, event_header, 32);

    Alg cspadFexAlg("cspadFexAlg", 1, 2, 3);
    unsigned segment = 0;
    Names& fexNames = *new(dgram.xtc) Names("xppcspad", cspadFexAlg, "cspad", "detnum1234", _src, segment);
    fexNames.add(dgram.xtc, myFexDef);
    m_namesVec.push_back(NameIndex(fexNames));

    Alg cspadRawAlg("cspadRawAlg", 1, 2, 3);
    Names& rawNames = *new(dgram.xtc) Names("xppcspad", cspadRawAlg, "cspad", "detnum1234", _src, segment);
    rawNames.add(dgram.xtc, myRawDef);
    m_namesVec.push_back(NameIndex(rawNames));
}

AreaDetector::AreaDetector(unsigned src) : Detector(src), m_evtcount(0) {}

void AreaDetector::event(Dgram& dgram, PGPData* pgp_data)
{
    m_evtcount+=1;
    int index = __builtin_ffs(pgp_data->buffer_mask) - 1;
    Transition* event_header = reinterpret_cast<Transition*>(pgp_data->buffers[index].data);

    unsigned nameId=0;
    CreateData fex(dgram.xtc, m_namesVec, nameId, _src);
    unsigned shape[MaxRank] = {3,3};
    Array<uint16_t> arrayT = fex.allocate<uint16_t>(FexDef::array_fex,shape);
    uint16_t* rawdata = (uint16_t*)(event_header+1);
    for(unsigned i=0; i<shape[0]; i++){
        for (unsigned j=0; j<shape[1]; j++) {
            arrayT(i,j) = i+j;
        }
    }
    
    // raw data
    memcpy(&dgram, event_header, 32);
    nameId = 1;
    DescribedData raw(dgram.xtc, m_namesVec, nameId, _src);
    unsigned size = 0;
    unsigned nlanes = 0;
    for (int l=0; l<8; l++) {
        if (pgp_data->buffer_mask & (1 << l)) {
            // size without Event header
            int data_size = pgp_data->buffers[l].size - 32;
            memcpy((uint8_t*)raw.data() + size, (uint8_t*)pgp_data->buffers[l].data + 32, data_size);
            size += data_size;
            nlanes++;
         }
     }
    raw.set_data_length(size);
    unsigned raw_shape[MaxRank] = {nlanes, size / nlanes / 2};
    raw.set_array_shape(RawDef::array_raw, raw_shape);
}
