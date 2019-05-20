
#include "AreaDetector.hh"
#include "TimingHeader.hh"
#include "AxisDriver.h"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"

using namespace XtcData;
using json = nlohmann::json;

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


AreaDetector::AreaDetector(Parameters* para, MemPool* pool) :
    Detector(para, pool), m_evtcount(0)
{
}

json AreaDetector::connectionInfo()
{
    int fd = open(m_para->device.c_str(), O_RDWR);
    if (fd < 0) {
        std::cout<<"Error opening "<< m_para->device << '\n';
        return json();
    }
    uint32_t reg;
    dmaReadRegister(fd, 0x00a00008, &reg);
    close(fd);
    int x = (reg >> 16) & 0xFF;
    int y = (reg >> 8) & 0xFF;
    int port = reg & 0xFF;
    std::string xpmIp = {"10.0." + std::to_string(x) + '.' + std::to_string(y)};
    json info = {{"xpm_ip", xpmIp}, {"xpm_port", port}};
    return info;
}

// setup up fake cameras to receive data over pgp
void AreaDetector::connect(const json& json)
{
    std::cout<<"AreaDetector connect\n";
    // FIXME make configureable
    int length = 100;
    int links = m_para->laneMask;

    int fd = open(m_para->device.c_str(), O_RDWR);
    if (fd < 0) {
        std::cout<<"Error opening "<< m_para->device << '\n';
        return;
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

unsigned AreaDetector::configure(Xtc& xtc)
{
    printf("AreaDetector configure\n");

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
