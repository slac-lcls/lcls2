
#include "AreaDetector.hh"
#include "TimingHeader.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"

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
    Pds::TimingHeader* timing_header = reinterpret_cast<Pds::TimingHeader*>(pgp_data->buffers[index].data);
    dgram.seq = timing_header->seq;
    dgram.env = timing_header->env;

    Alg cspadFexAlg("cspadFexAlg", 1, 2, 3);
    unsigned segment = 0;
    NamesId fexNamesId(m_nodeId,FexNamesIndex);
    Names& fexNames = *new(dgram.xtc) Names("xppcspad", cspadFexAlg, "cspad", "detnum1234", fexNamesId, segment);
    fexNames.add(dgram.xtc, myFexDef);
    m_namesVec[fexNamesId] = NameIndex(fexNames);

    Alg cspadRawAlg("cspadRawAlg", 1, 2, 3);
    NamesId rawNamesId(m_nodeId,RawNamesIndex);
    Names& rawNames = *new(dgram.xtc) Names("xppcspad", cspadRawAlg, "cspad", "detnum1234", rawNamesId, segment);
    rawNames.add(dgram.xtc, myRawDef);
    m_namesVec[rawNamesId] = NameIndex(rawNames);
}

AreaDetector::AreaDetector(unsigned nodeId) : Detector(nodeId), m_evtcount(0) {}

void AreaDetector::event(Dgram& dgram, PGPData* pgp_data)
{
    m_evtcount+=1;
    int index = __builtin_ffs(pgp_data->buffer_mask) - 1;

    Pds::TimingHeader* timing_header = reinterpret_cast<Pds::TimingHeader*>(pgp_data->buffers[index].data);
    dgram.seq = timing_header->seq;
    dgram.env = timing_header->env;

    // fex data
    NamesId fexNamesId(m_nodeId,FexNamesIndex);
    CreateData fex(dgram.xtc, m_namesVec, fexNamesId);
    unsigned shape[MaxRank] = {3,3};
    Array<uint16_t> arrayT = fex.allocate<uint16_t>(FexDef::array_fex,shape);
    uint16_t* rawdata = (uint16_t*)(timing_header+1);
    for(unsigned i=0; i<shape[0]; i++){
        for (unsigned j=0; j<shape[1]; j++) {
            arrayT(i,j) = i+j;
        }
    }
    
    // raw data
    NamesId rawNamesId(m_nodeId,RawNamesIndex);
    DescribedData raw(dgram.xtc, m_namesVec, rawNamesId);
    unsigned size = 0;
    unsigned nlanes = 0;
    for (int l=0; l<8; l++) {
        if (pgp_data->buffer_mask & (1 << l)) {
            // size without Event header
            int data_size = pgp_data->buffers[l].size - 32;
            memcpy((uint8_t*)raw.data() + size, (uint8_t*)rawdata, data_size);
            size += data_size;
            nlanes++;
         }
     }
    raw.set_data_length(size);
    unsigned raw_shape[MaxRank] = {nlanes, size / nlanes / 2};
    raw.set_array_shape(RawDef::array_raw, raw_shape);
}
