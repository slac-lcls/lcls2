#include "Wave8.hh"
#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psdaq/service/Json2Xtc.hh"
#include "rapidjson/document.h"
#include "xtcdata/xtc/XtcIterator.hh"
#include "AxisDriver.h"

#include <fcntl.h>
#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

using namespace XtcData;
using namespace rapidjson;

using json = nlohmann::json;

namespace Drp {

class W8Def : public VarDef
{
public:
    enum index {
        data
    };
    W8Def()
    {
        Alg alg("raw", 1, 2, 3);
        NameVec.push_back({"data", Name::UINT8, 1});
    }
} W8Def;

Wave8::Wave8(Parameters* para, MemPool* pool) :
    XpmDetector(para, pool),
    m_evtNamesId(-1, -1), // placeholder
    m_connect_json("")
{
}

static void check(PyObject* obj) {
    if (!obj) {
        PyErr_Print();
        throw "**** python error\n";
    }
}

void Wave8::_addJson(Xtc& xtc, NamesId& configNamesId, const std::string& config_alias) {

    // returns new reference
    PyObject* pModule = PyImport_ImportModule("psalg.configdb.w8_config");
    check(pModule);
    // returns borrowed reference
    PyObject* pDict = PyModule_GetDict(pModule);
    check(pDict);
    // returns borrowed reference
    PyObject* pFunc = PyDict_GetItemString(pDict, (char*)"w8_config");
    check(pFunc);
    // returns new reference
    PyObject* mybytes = PyObject_CallFunction(pFunc, "sss", m_connect_json.c_str(), config_alias.c_str(), m_para->detName.c_str());
    check(mybytes);

    // returns new reference
    PyObject * json_bytes = PyUnicode_AsASCIIString(mybytes);
    check(json_bytes);
    char* json = (char*)PyBytes_AsString(json_bytes);

    // convert to json to xtc
    const unsigned BUFSIZE = 1024*1024;
    char buffer[BUFSIZE];
    unsigned len = Pds::translateJson2Xtc(json, buffer, configNamesId);
    if (len>BUFSIZE) {
        throw "**** Config json output too large for buffer\n";
    }
    if (len <= 0) {
        throw "**** Config json translation error\n";
    }

    // append the config xtc info to the dgram
    Xtc& jsonxtc = *(Xtc*)buffer;
    memcpy(xtc.next(),jsonxtc.payload(),jsonxtc.sizeofPayload());
    xtc.alloc(jsonxtc.sizeofPayload());

    Py_DECREF(pModule);
    Py_DECREF(mybytes);
    Py_DECREF(json_bytes);
}

void Wave8::connect(const json& connect_json, const std::string& collectionId)
{
  m_connect_json = connect_json.dump();
}

unsigned Wave8::configure(const std::string& config_alias, Xtc& xtc)
{
    m_evtNamesId = NamesId(nodeId, EventNamesIndex);
    // set up the names for the configuration data
    NamesId configNamesId(nodeId,ConfigNamesIndex);
    _addJson(xtc, configNamesId, config_alias);

    // set up the names for L1Accept data
    Alg w8Alg("w8", 1, 2, 3); // TODO: should this be configured by w8config.py?
    Names& eventNames = *new(xtc) Names(m_para->detName.c_str(), w8Alg, "w8", "detnum1235", m_evtNamesId, m_para->detSegment);
    eventNames.add(xtc, W8Def);
    m_namesLookup[m_evtNamesId] = NameIndex(eventNames);
    return 0;
}

void Wave8::event(XtcData::Dgram& dgram, PGPEvent* event)
{
    CreateData w8(dgram.xtc, m_namesLookup, m_evtNamesId);

    int lane = __builtin_ffs(event->mask) - 1;
    // there should be only one lane of data in the timing system
    uint32_t dmaIndex = event->buffers[lane].index;
    unsigned data_size = event->buffers[lane].size - sizeof(Pds::TimingHeader);
    unsigned shape[MaxRank];
    shape[0] = data_size;
    Array<uint8_t> arrayT = w8.allocate<uint8_t>(W8Def::data, shape);
    memcpy(arrayT.data(), (uint8_t*)m_pool->dmaBuffers[dmaIndex] + sizeof(Pds::TimingHeader), data_size);
}

}
