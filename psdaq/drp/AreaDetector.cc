
#include "AreaDetector.hh"
#include "TimingHeader.hh"
#include "AxisDriver.h"
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


AreaDetector::AreaDetector(Parameters* para) :
    Detector(para), m_evtcount(0)
{
}

// setup up fake cameras to receive data over pgp
void AreaDetector::connect()
{
    std::cout<<"AreaDetector connect\n";
    // FIXME make configureable
    int length = 500;
    int links = 0xf;

    int fd = open("/dev/datadev_1", O_RDWR);
    if (fd < 0) {
        std::cout<<"Error opening /dev/datadev_1\n";
    }
    int partition = m_para->partition;
    uint32_t v = ((partition&0xf)<<0) |
                  ((length&0xffffff)<<4) |
                  (links<<28);
    dmaWriteRegister(fd, 0x00a00000, v);
    uint32_t w;
    dmaReadRegister(fd, 0x00a00000, &w);
    printf("Configured partition [%u], length [%u], links [%x]: [%x](%x)\n",
           partition, length, links, v, w);
    for (unsigned i=0; i<4; i++) {
        if (links&(1<<i)) {
            dmaWriteRegister(fd, 0x00800084+32*i, 0x1f00);
        }
    }
    close(fd);
}

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
    Names& fexNames = *new(dgram.xtc) Names("xppcspad", cspadFexAlg, "cspad",
                                            "detnum1234", fexNamesId, segment);
    fexNames.add(dgram.xtc, myFexDef);
    m_namesLookup[fexNamesId] = NameIndex(fexNames);

    Alg cspadRawAlg("cspadRawAlg", 1, 2, 3);
    NamesId rawNamesId(m_nodeId,RawNamesIndex);
    Names& rawNames = *new(dgram.xtc) Names("xppcspad", cspadRawAlg, "cspad",
                                            "detnum1234", rawNamesId, segment);
    rawNames.add(dgram.xtc, myRawDef);
    m_namesLookup[rawNamesId] = NameIndex(rawNames);
}

void AreaDetector::event(Dgram& dgram, PGPData* pgp_data)
{
    m_evtcount+=1;
    int index = __builtin_ffs(pgp_data->buffer_mask) - 1;
    Pds::TimingHeader* timing_header = reinterpret_cast<Pds::TimingHeader*>(pgp_data->buffers[index].data);
    dgram.seq = timing_header->seq;
    dgram.env = timing_header->env;

    // fex data
    NamesId fexNamesId(m_nodeId,FexNamesIndex);
    CreateData fex(dgram.xtc, m_namesLookup, fexNamesId);
    unsigned shape[MaxRank] = {3,3};
    Array<uint16_t> arrayT = fex.allocate<uint16_t>(FexDef::array_fex,shape);
    uint32_t* rawdata = (uint32_t*)(timing_header+1);

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
    NamesId rawNamesId(m_nodeId,RawNamesIndex);
    DescribedData raw(dgram.xtc, m_namesLookup, rawNamesId);
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
