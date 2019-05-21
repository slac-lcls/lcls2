#include "TimingSystem.hh"
#include "TimingHeader.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "xtcdata/xtc/Json2Xtc.hh"
#include "rapidjson/document.h"
#include "xtcdata/xtc/XtcIterator.hh"

#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

using namespace XtcData;
using namespace rapidjson;

using json = nlohmann::json;

namespace Drp {

class TSDef : public VarDef
{
public:
    enum index {
        dataval
    };
    TSDef()
    {
        Alg alg("raw", 1, 2, 3);
        NameVec.push_back({"dataval", Name::UINT32, 0});
    }
} TSDef;

TimingSystem::TimingSystem(Parameters* para, MemPool* pool) :
    XpmDetector(para, pool),
    m_evtcount(0),
    m_evtNamesId(nodeId, EventNamesIndex)
{
}

static void check(PyObject* obj) {
    if (!obj) {
        PyErr_Print();
        throw "**** python error\n";
    }
}

void TimingSystem::_addJson(Xtc& xtc, NamesId& configNamesId) {

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
    PyObject* mybytes = PyObject_CallFunction(pFunc,"ssssss","DAQ:LAB2:HSD:DEV02",
                                              "mcbrowne:psana@psdb-dev:9306",
                                              "configDB", "TMO", "BEAM",
                                              "xpphsd");
    check(mybytes);
    // returns new reference
    PyObject * json_bytes = PyUnicode_AsASCIIString(mybytes);
    check(json_bytes);
    char* json = (char*)PyBytes_AsString(json_bytes);
    printf("json: %s\n",json);

    // convert to json to xtc
    const unsigned BUFSIZE = 1024*1024;
    char buffer[BUFSIZE];
    unsigned len = translateJson2Xtc(json, buffer, configNamesId);
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

// TODO: put timeout value in connect and attach (conceptually like Collection.cc CollectionApp::handlePlat)

void TimingSystem::connect(const json& json_connect_info)
{
    printf("*** here in connect\n");
    XpmDetector::connect(json_connect_info);
    // returns new reference
    PyObject* pModule = PyImport_ImportModule("psalg.configdb.ts_connect");
    check(pModule);
    // returns borrowed reference
    PyObject* pDict = PyModule_GetDict(pModule);
    check(pDict);
    // returns borrowed reference
    PyObject* pFunc = PyDict_GetItemString(pDict, (char*)"ts_connect");
    check(pFunc);
    // returns new reference
    PyObject* mybytes = PyObject_CallFunction(pFunc,"s",json_connect_info.dump().c_str());
    check(mybytes);
    // returns new reference
    PyObject * json_bytes = PyUnicode_AsASCIIString(mybytes);
    check(json_bytes);
    char* json = (char*)PyBytes_AsString(json_bytes);
    printf("json: %s\n",json);

    Py_DECREF(pModule);
    Py_DECREF(mybytes);
    Py_DECREF(json_bytes);
}

unsigned TimingSystem::configure(Xtc& xtc)
{
    printf("*** here in config\n");
    // set up the names for the configuration data
    NamesId configNamesId(nodeId,ConfigNamesIndex);
    TimingSystem::_addJson(xtc, configNamesId);

    // set up the names for L1Accept data
    Alg tsAlg("ts", 1, 2, 3); // TODO: should this be configured by tsconfig.py?
    unsigned segment = 0;
    Names& eventNames = *new(xtc) Names("xppts", tsAlg, "ts", "detnum1235", m_evtNamesId, segment);
    eventNames.add(xtc, TSDef);
    m_namesLookup[m_evtNamesId] = NameIndex(eventNames);
    return 0;
    printf("*** done config\n");
}

void TimingSystem::event(XtcData::Dgram& dgram, PGPEvent* event)
{
    printf("*** here in event\n");
    m_evtcount+=1;
    CreateData ts(dgram.xtc, m_namesLookup, m_evtNamesId);

    ts.set_value(TSDef::dataval, (uint32_t) 3);
    printf("*** done event\n");
}

}
