#include "TimingSystem.hh"
#include "TimingHeader.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "xtcdata/xtc/Json2Xtc.hh"
#include "rapidjson/document.h"
#include "xtcdata/xtc/XtcIterator.hh"
#include "AxisDriver.h"

#include <fcntl.h>
#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

const unsigned BUFSIZE = 1024*1024*32;
static char config_buf [BUFSIZE];

using namespace XtcData;
using namespace rapidjson;

using json = nlohmann::json;

namespace Drp {

class TSDef : public VarDef
{
public:
    enum index {
        data
    };
    TSDef()
    {
        Alg alg("raw", 1, 2, 3);
        NameVec.push_back({"data", Name::UINT8, 1});
    }
} TSDef;

TimingSystem::TimingSystem(Parameters* para, MemPool* pool) :
    XpmDetector(para, pool),
    m_evtcount(0),
    m_evtNamesId(nodeId, EventNamesIndex),
    m_connect_json("")
{
}

static void check(PyObject* obj) {
    if (!obj) {
        PyErr_Print();
        throw "**** python error\n";
    }
}

void TimingSystem::_addJson(Xtc& xtc, NamesId& configNamesId, const std::string& config_alias) {

    // returns new reference
    PyObject* pModule = PyImport_ImportModule("psalg.configdb.ts_config");
    check(pModule);
    // returns borrowed reference
    PyObject* pDict = PyModule_GetDict(pModule);
    check(pDict);
    // returns borrowed reference
    PyObject* pFunc = PyDict_GetItemString(pDict, (char*)"ts_config");
    check(pFunc);
    // need to get the dbase connection info via collection
    // returns new reference
    PyObject* mybytes = PyObject_CallFunction(pFunc,"sss",m_connect_json.c_str(), config_alias.c_str(), m_para->detName.c_str());
    check(mybytes);
    // returns new reference
    PyObject * json_bytes = PyUnicode_AsASCIIString(mybytes);
    check(json_bytes);
    char* json = (char*)PyBytes_AsString(json_bytes);

    // convert to json to xtc
    unsigned len = translateJson2Xtc(json, config_buf, configNamesId);
    if (len>BUFSIZE) {
        throw "**** Config json output too large for buffer\n";
    }
    if (len <= 0) {
        throw "**** Config json translation error\n";
    }

    // append the config xtc info to the dgram
    Xtc& jsonxtc = *(Xtc*)config_buf;
    memcpy(xtc.next(),jsonxtc.payload(),jsonxtc.sizeofPayload());
    xtc.alloc(jsonxtc.sizeofPayload());

    Py_DECREF(pModule);
    Py_DECREF(mybytes);
    Py_DECREF(json_bytes);

}

// TODO: put timeout value in connect and attach (conceptually like Collection.cc CollectionApp::handlePlat)

void TimingSystem::connect(const json& connect_json, const std::string& collectionId)
{
    XpmDetector::connect(connect_json, collectionId);
    
    int fd = open(m_para->device.c_str(), O_RDWR);
    if (fd < 0) {
        std::cout<<"Error opening "<< m_para->device << '\n';
        return;
    }

    uint32_t val;
    dmaReadRegister(fd, 0x00a00000, &val);
    // zero out the "length" field which changes the behaviour of the
    // firmware from fake-camera mode to timing-system mode
    val&=0xf000000f;
    dmaWriteRegister(fd, 0x00a00000, val);
    close(fd);

    // returns new reference
    PyObject* pModule = PyImport_ImportModule("psalg.configdb.ts_connect");
    check(pModule);
    // returns borrowed reference
    PyObject* pDict = PyModule_GetDict(pModule);
    check(pDict);
    // returns borrowed reference
    PyObject* pFunc = PyDict_GetItemString(pDict, (char*)"ts_connect");
    check(pFunc);
    m_connect_json = connect_json.dump();
    // returns new reference
    PyObject* mybytes = PyObject_CallFunction(pFunc,"s",m_connect_json.c_str());
    check(mybytes);

    Py_DECREF(pModule);
    Py_DECREF(mybytes);
}

unsigned TimingSystem::configure(const std::string& config_alias, Xtc& xtc)
{
    // set up the names for the configuration data
    NamesId configNamesId(nodeId,ConfigNamesIndex);
    _addJson(xtc, configNamesId, config_alias);

    // set up the names for L1Accept data
    Alg tsAlg("ts", 1, 2, 3); // TODO: should this be configured by tsconfig.py?
    Names& eventNames = *new(xtc) Names(m_para->detName.c_str(), tsAlg, "ts", "detnum1235", m_evtNamesId, m_para->detSegment);
    eventNames.add(xtc, TSDef);
    m_namesLookup[m_evtNamesId] = NameIndex(eventNames);
    return 0;
}

void TimingSystem::event(XtcData::Dgram& dgram, PGPEvent* event)
{
    m_evtcount+=1;

    CreateData ts(dgram.xtc, m_namesLookup, m_evtNamesId);

    int lane = __builtin_ffs(event->mask) - 1;
    // there should be only one lane of data in the timing system
    uint32_t dmaIndex = event->buffers[lane].index;
    unsigned data_size = event->buffers[lane].size - sizeof(Pds::TimingHeader);
    unsigned shape[MaxRank];
    shape[0] = data_size;
    Array<uint8_t> arrayT = ts.allocate<uint8_t>(TSDef::data, shape);
    memcpy(arrayT.data(), (uint8_t*)m_pool->dmaBuffers[dmaIndex] + sizeof(Pds::TimingHeader), data_size);
}

}
