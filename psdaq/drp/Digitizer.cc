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
#include "Si570.hh"
#include "psalg/utils/SysLog.hh"

#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <fstream>
#include <unistd.h>
#include <netdb.h>
#include <arpa/inet.h>

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
    Detector    (para, pool),
    m_evtNamesId(-1, -1), // placeholder
    m_epics_name(para->kwargs["hsd_epics_prefix"]),
    m_paddr     (_getPaddr())
{
    printf("*** found epics name %s\n",m_epics_name.c_str());
}

static void check(PyObject* obj) {
    if (!obj) {
        PyErr_Print();
        throw "**** python error";
    }
}

unsigned Digitizer::_getPaddr()
{
    // Check PGP reference clock, reprogram if necessary
    int fd = open(m_para->device.c_str(), O_RDWR);
    AxiVersion vsn;
    axiVersionGet(fd, &vsn);
    if (vsn.userValues[2] == 0) {  // Only one PCIe interface has access to I2C bus
      unsigned pgpclk;
      dmaReadRegister(fd, 0x80010c, &pgpclk);
      printf("PGP RefClk %f MHz\n", double(pgpclk&0x1fffffff)*1.e-6);
      if ((pgpclk&0x1fffffff) < 185000000) { // target is 185.7 MHz
        //  Set the I2C Mux
        dmaWriteRegister(fd, 0x00e00000, (1<<2));
        //  Configure the Si570
        Si570 s(fd, 0x00e00800);
        s.program();
      }
    }

    { struct addrinfo hints;
      struct addrinfo* result;

      memset(&hints, 0, sizeof(struct addrinfo));
      hints.ai_family = AF_INET;       /* Allow IPv4 or IPv6 */
      hints.ai_socktype = SOCK_DGRAM; /* Datagram socket */
      hints.ai_flags = AI_PASSIVE;    /* For wildcard IP address */

      char hname[64];
      gethostname(hname,64);
      int s = getaddrinfo(hname, NULL, &hints, &result);
      if (s != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(s));
        exit(EXIT_FAILURE);
      }

      sockaddr_in* saddr = (sockaddr_in*)result->ai_addr;


      unsigned id = 0xfb000000 |
        (ntohl(saddr->sin_addr.s_addr)&0xffff);

      for(unsigned i=0; i<4; i++) {
        unsigned link = (vsn.userValues[2] == 0) ? i : i+4;
        dmaWriteRegister(fd, 0x00a40010+4*(link&3), id | (link<<16));
      }
    }

    close(fd);

    // returns new reference
    PyObject* pModule = PyImport_ImportModule("psdaq.configdb.hsd_connect");
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
        logging::critical("Parse error: %s, location %zu",
                          GetParseError_En(d->GetParseError()), d->GetErrorOffset());
        throw "Parse error";
    }
    const Value& a = (*d)["paddr"];

    unsigned reg = a.GetInt();
    if (!reg) {
        const char msg[] = "XPM Remote link id register is zero";
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

unsigned Digitizer::_addJson(Xtc& xtc, NamesId& configNamesId, const std::string& config_alias) {

  timespec tv_b; clock_gettime(CLOCK_REALTIME,&tv_b);

#define CHECK_TIME(s) {                                                 \
    timespec tv; clock_gettime(CLOCK_REALTIME,&tv);                     \
    printf("%s %f seconds\n",#s,                                        \
           double(tv.tv_sec-tv_b.tv_sec)+1.e-9*(double(tv.tv_nsec)-double(tv_b.tv_nsec))); }


    // returns new reference
    PyObject* pModule = PyImport_ImportModule("psdaq.configdb.hsd_config");
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
    PyObject* mybytes = PyObject_CallFunction(pFunc,"ssssii",
                                              m_connect_json.c_str(),
                                              m_epics_name.c_str(),
                                              config_alias.c_str(),
                                              m_para->detName.c_str(),
                                              m_para->detSegment,
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
    unsigned len = Pds::translateJson2Xtc(json, buffer, configNamesId, m_para->detName.c_str(), m_para->detSegment);
    if (len>BUFSIZE) {
        throw "**** Config json output too large for buffer";
    }
    if (len <= 0) {
        throw "**** Config json translation error";
    }

    CHECK_TIME(translateJson);

    // append the config xtc info to the dgram
    Xtc& jsonxtc = *(Xtc*)buffer;
    memcpy((void*)xtc.next(),(const void*)jsonxtc.payload(),jsonxtc.sizeofPayload());
    xtc.alloc(jsonxtc.sizeofPayload());

    // get the lane mask from the json
    unsigned lane_mask = 1;
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
    lane_mask = Digitizer::_addJson(xtc, configNamesId, config_alias);

    // set up the names for L1Accept data
    Alg alg("raw", 2, 0, 0);
    Names& eventNames = *new(xtc) Names(m_para->detName.c_str(), alg,
                                        m_para->detType.c_str(), m_para->serNo.c_str(), m_evtNamesId, m_para->detSegment);
    HsdDef myHsdDef(lane_mask);
    eventNames.add(xtc, myHsdDef);
    m_namesLookup[m_evtNamesId] = NameIndex(eventNames);
    return 0;
}

void Digitizer::event(XtcData::Dgram& dgram, PGPEvent* event)
{
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

    if ((timing_header->_opaque[1] & (1<<31))==0)  // check JESD status bit
      dgram.xtc.damage.increase(Damage::UserDefined);

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

void Digitizer::shutdown()
{
    // returns new reference
    PyObject* pModule = PyImport_ImportModule("psdaq.configdb.hsd_config");
    check(pModule);

    // returns borrowed reference
    PyObject* pDict = PyModule_GetDict(pModule);
    check(pDict);
    // returns borrowed reference
    PyObject* pFunc = PyDict_GetItemString(pDict, (char*)"hsd_unconfig");
    check(pFunc);

    // returns new reference
    PyObject_CallFunction(pFunc,"s",
                          m_epics_name.c_str());
    Py_DECREF(pModule);
}

}
