#include "Digitizer.hh"
#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psdaq/service/Json2Xtc.hh"
#include "rapidjson/document.h"
#include "xtcdata/xtc/XtcIterator.hh"
#include "psalg/digitizer/Hsd.hh"
#include "DataDriver.h"
#include "psalg/utils/SysLog.hh"

#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <fstream>

using namespace XtcData;
using namespace rapidjson;
using logging = psalg::SysLog;

using json = nlohmann::json;

namespace Drp {

class HsdDef : public VarDef
{
public:
    enum index {EventHeader = 0};
    HsdDef(unsigned lane_mask)
    {
        Alg alg("fpga", 1, 2, 3);
        const unsigned nameLen = 7;
        char chanName[nameLen];
        NameVec.push_back({"eventHeader", Name::UINT32, 1});
        for (unsigned i = 0; i < sizeof(lane_mask)*sizeof(uint8_t); i++){
            if (!((1<<i)&lane_mask)) continue;
            snprintf(chanName, nameLen, "chan%2.2d", i);
            NameVec.push_back({chanName, alg});
        }
    }
};

Digitizer::Digitizer(Parameters* para, MemPool* pool) :
    Detector(para, pool),
    m_evtcount(0),
    m_evtNamesId(-1, -1), // placeholder
    m_epics_name(para->kwargs["hsd_epics_prefix"]),
    m_paddr     (_getPaddr())
{
    printf("*** found epics name %s\n",m_epics_name.c_str());
}

static void check(PyObject* obj) {
    if (!obj) {
        PyErr_Print();
        throw "**** python error\n";
    }
}

unsigned Digitizer::_getPaddr()
{
    // returns new reference
    PyObject* pModule = PyImport_ImportModule("psalg.configdb.hsd_connect");
    check(pModule);

    // returns borrowed reference
    PyObject* pDict = PyModule_GetDict(pModule);
    check(pDict);
    // returns borrowed reference
    PyObject* pFunc = PyDict_GetItemString(pDict, (char*)"hsd_connect");
    check(pFunc);

    // returns new reference
    PyObject* mybytes = PyObject_CallFunction(pFunc,"s",
                                              m_epics_name.c_str());

    check(mybytes);

    PyObject * json_bytes = PyUnicode_AsASCIIString(mybytes);
    check(json_bytes);
    char* json_str = (char*)PyBytes_AsString(json_bytes);

    Document *d = new Document();
    d->Parse(json_str);
    if (d->HasParseError()) {
        printf("Parse error: %s, location %zu\n",
               GetParseError_En(d->GetParseError()), d->GetErrorOffset());
        abort();
    }
    const Value& a = (*d)["paddr"];

    unsigned reg = a.GetInt();
    if (!reg) {
        const char msg[] = "XPM Remote link id register is zero\n";
        logging::error("%s", msg);
        throw msg;
    }

    Py_DECREF(pModule);
    Py_DECREF(mybytes);
    Py_DECREF(json_bytes);

    return reg;
}

json Digitizer::connectionInfo()
{
    unsigned reg = m_paddr;
    int xpm  = (reg >> 20) & 0xF;
    int port = (reg >>  0) & 0xFF;
    json info = {{"xpm_id", xpm}, {"xpm_port", port}};
    return info;
}

unsigned Digitizer::_addJson(Xtc& xtc, NamesId& configNamesId) {

  timespec tv_b; clock_gettime(CLOCK_REALTIME,&tv_b);

#define CHECK_TIME(s) {                                                 \
    timespec tv; clock_gettime(CLOCK_REALTIME,&tv);                     \
    printf("%s %f seconds\n",#s,                                        \
           double(tv.tv_sec-tv_b.tv_sec)+1.e-9*(double(tv.tv_nsec)-double(tv_b.tv_nsec))); }


    // returns new reference
    PyObject* pModule = PyImport_ImportModule("psalg.configdb.hsd_config");
    check(pModule);

    CHECK_TIME(PyImport);

    // returns borrowed reference
    PyObject* pDict = PyModule_GetDict(pModule);
    check(pDict);
    // returns borrowed reference
    PyObject* pFunc = PyDict_GetItemString(pDict, (char*)"hsd_config");
    check(pFunc);

    CHECK_TIME(PyDict_Get);

    // returns new reference
    PyObject* mybytes = PyObject_CallFunction(pFunc,"ssssi",
                                              m_connect_json.c_str(),
                                              m_epics_name.c_str(),
                                              "BEAM", 
                                              m_para->detName.c_str(),
                                              m_readoutGroup);

    CHECK_TIME(PyObj_Call);

    check(mybytes);
    // returns new reference
    PyObject * json_bytes = PyUnicode_AsASCIIString(mybytes);
    check(json_bytes);
    char* json = (char*)PyBytes_AsString(json_bytes);
    printf("json: %s\n",json);

    // convert to json to xtc
    const unsigned BUFSIZE = 1024*1024;
    char buffer[BUFSIZE];
    unsigned len = Pds::translateJson2Xtc(json, buffer, configNamesId, m_para->detSegment);
    if (len>BUFSIZE) {
        throw "**** Config json output too large for buffer\n";
    }
    if (len <= 0) {
        throw "**** Config json translation error\n";
    }

    CHECK_TIME(translateJson);

    // append the config xtc info to the dgram
    Xtc& jsonxtc = *(Xtc*)buffer;
    memcpy(xtc.next(),jsonxtc.payload(),jsonxtc.sizeofPayload());
    xtc.alloc(jsonxtc.sizeofPayload());

    // get the lane mask from the json
    unsigned lane_mask = 0;
    Document top;
    if (top.Parse(json).HasParseError())
        fprintf(stderr,"*** json parse error\n");
    else {
      const Value& enable = top["paddr"];
      std::string enable_type = top[":types:"]["enable"][0].GetString();
      unsigned length = top[":types:"]["enable"][1].GetInt();
      for (unsigned i=0; i<length; i++) if (enable[i].GetInt()) lane_mask |= 1<< i;
    }
    lane_mask = 1; // override temporarily!
    printf("hsd lane_mask is 0x%x\n",lane_mask);

    Py_DECREF(pModule);
    Py_DECREF(mybytes);
    Py_DECREF(json_bytes);

    CHECK_TIME(Done);

    return lane_mask;
}

void Digitizer::connect(const json& connect_json, const std::string& collectionId)
{
  m_connect_json = connect_json.dump();
  m_readoutGroup = connect_json["body"]["drp"][collectionId]["det_info"]["readout"];
}

unsigned Digitizer::configure(const std::string& config_alias, Xtc& xtc)
{
    //  Reset the PGP links
    int fd = open(m_para->device.c_str(), O_RDWR);
    //  user reset
    dmaWriteRegister(fd, 0x00800000, (1<<31));
    usleep(10);
    dmaWriteRegister(fd, 0x00800000, 0);
    //  QPLL reset
    dmaWriteRegister(fd, 0x00a40024, 1);
    usleep(10);
    dmaWriteRegister(fd, 0x00a40024, 0);
    usleep(10);
    //  Reset the Tx and Rx
    dmaWriteRegister(fd, 0x00a40024, 6);
    usleep(10);
    dmaWriteRegister(fd, 0x00a40024, 0);
    close(fd);

    unsigned lane_mask;
    m_evtNamesId = NamesId(nodeId, EventNamesIndex);
    // set up the names for the configuration data
    NamesId configNamesId(nodeId,ConfigNamesIndex);
    lane_mask = Digitizer::_addJson(xtc, configNamesId);

    // set up the names for L1Accept data
    Alg hsdAlg("hsd", 1, 2, 3);
    Names& eventNames = *new(xtc) Names(m_para->detName.c_str(), hsdAlg, "hsd", "detnum1235", m_evtNamesId, m_para->detSegment);
    HsdDef myHsdDef(lane_mask);
    eventNames.add(xtc, myHsdDef);
    m_namesLookup[m_evtNamesId] = NameIndex(eventNames);
    return 0;
}

void Digitizer::event(XtcData::Dgram& dgram, PGPEvent* event)
{
    m_evtcount+=1;
    CreateData hsd(dgram.xtc, m_namesLookup, m_evtNamesId);

    // HSD data includes two uint32_t "event header" words
    unsigned data_size;
    unsigned shape[MaxRank];
    shape[0] = 2;
    Array<uint32_t> arrayH = hsd.allocate<uint32_t>(HsdDef::EventHeader, shape);

    int lane = __builtin_ffs(event->mask) - 1;
    uint32_t dmaIndex = event->buffers[lane].index;
    Pds::TimingHeader* timing_header = (Pds::TimingHeader*)m_pool->dmaBuffers[dmaIndex];
    arrayH(0) = timing_header->_opaque[0];
    arrayH(1) = timing_header->_opaque[1];
    for (int i=0; i<4; i++) {
        if (event->mask & (1 << i)) {
            data_size = event->buffers[i].size - sizeof(Pds::TimingHeader);
            shape[0] = data_size;
            Array<uint8_t> arrayT = hsd.allocate<uint8_t>(i+1, shape);
            uint32_t dmaIndex = event->buffers[i].index;
            memcpy(arrayT.data(), (uint8_t*)m_pool->dmaBuffers[dmaIndex] + sizeof(Pds::TimingHeader), data_size);
            // example showing how to use psalg Hsd code to extract data
            Pds::HSD::Channel channel(&m_allocator,
                                      (const uint32_t*)arrayH.data(),
                                      (const uint8_t*)arrayT.data());
            //printf("*** npeaks %d\n",channel.npeaks());
         }
    }
}
  
}
