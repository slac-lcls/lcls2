#include "BEBDetector.hh"
#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psdaq/service/Json2Xtc.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include "psalg/utils/SysLog.hh"
#include "AxisDriver.h"

#include <thread>
#include <fcntl.h>
#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

using namespace XtcData;
using namespace rapidjson;
using logging = psalg::SysLog;

using json = nlohmann::json;

namespace Drp {

PyObject* BEBDetector::_check(PyObject* obj) {
    if (!obj) {
        PyErr_Print();
        throw "**** python error\n";
    }
    return obj;
}

std::string BEBDetector::_string_from_PyDict(PyObject* dict, const char* key)
{
    PyObject* item      = _check(PyDict_GetItemString(dict,key));
    PyObject * item_str = _check(PyUnicode_AsASCIIString(item));
    std::string s(PyBytes_AsString(item_str));
    Py_DECREF(item_str);
    return s;
}


BEBDetector::BEBDetector(Parameters* para, MemPool* pool) :
    Detector      (para, pool),
    m_connect_json(""),
    m_module      (0)
{
    virtChan = 1;
}

BEBDetector::~BEBDetector()
{
    Py_DECREF(m_module);
}

void BEBDetector::_init(const char* arg)
{
    char module_name[64];
    sprintf(module_name,"psdaq.configdb.%s_config",m_para->detType.c_str());

    char func_name[64];

    // returns new reference
    m_module = _check(PyImport_ImportModule(module_name));

    PyObject* pDict = _check(PyModule_GetDict(m_module));
    {
      sprintf(func_name,"%s_init",m_para->detType.c_str());
      PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)func_name));

      // returns new reference
#define MLOOKUP(m,name,dflt) (m.find(name)==m.end() ? dflt : m[name].c_str())
      const char* xpmpv = MLOOKUP(m_para->kwargs,"xpmpv",0);
      m_root = _check(PyObject_CallFunction(pFunc,"ss",arg,xpmpv));
    }

    {
      sprintf(func_name,"%s_connect",m_para->detType.c_str());
      PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)func_name));

      // returns new reference
      PyObject* mbytes = _check(PyObject_CallFunction(pFunc,"O",m_root));

      m_paddr = PyLong_AsLong(PyDict_GetItemString(mbytes, "paddr"));

      if (!m_paddr) {
        const char msg[] = "XPM Remote link id register is zero\n";
        logging::error("%s", msg);
        throw msg;
      }

      _connect(mbytes);

      Py_DECREF(mbytes);
    }
}

void BEBDetector::_init_feb()
{
    PyObject* pDict = _check(PyModule_GetDict(m_module));

    char func_name[64];
    sprintf(func_name,"%s_init_feb",m_para->detType.c_str());
    PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)func_name));

#define MLOOKUP(m,name,dflt) (m.find(name)==m.end() ? dflt : m[name].c_str())
    const char* lane = MLOOKUP(m_para->kwargs,"feb_lane"   ,0);
    const char* chan = MLOOKUP(m_para->kwargs,"feb_channel",0);
    PyObject* mybytes = _check(PyObject_CallFunction(pFunc, "ss", lane, chan));
    Py_DECREF(mybytes);
}

json BEBDetector::connectionInfo()
{
    unsigned reg = m_paddr;
    int xpm  = (reg >> 20) & 0x0F;
    int port = (reg >>  0) & 0xFF;
    json info = {{"xpm_id", xpm}, {"xpm_port", port}};
    return info;
}

unsigned BEBDetector::configure(const std::string& config_alias, 
                                Xtc&               xtc)
{
    PyObject* pDict = _check(PyModule_GetDict(m_module));

    char func_name[64];
    sprintf(func_name,"%s_config",m_para->detType.c_str());
    PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)func_name));

    // returns new reference
    PyObject* mybytes = _check(PyObject_CallFunction(pFunc, "Osssii",
                                                     m_root, m_connect_json.c_str(), config_alias.c_str(),
                                                     m_para->detName.c_str(), m_para->detSegment, m_readoutGroup));

    // returns new reference
    PyObject * json_bytes = _check(PyUnicode_AsASCIIString(mybytes));
    char* json = (char*)PyBytes_AsString(json_bytes);

    // convert json to xtc
    const int BUFSIZE = 1024*1024;
    char* buffer = new char[BUFSIZE];
    NamesId configNamesId(nodeId,ConfigNamesIndex);
    if (Pds::translateJson2Xtc(json, buffer, configNamesId, m_para->detName.c_str(), m_para->detSegment) < 0)
        throw "**** Config json translation error\n";

    Xtc& jsonxtc = *(Xtc*)buffer;
    if (jsonxtc.extent>BUFSIZE)
        throw "**** Config json output too large for buffer\n";

    XtcData::ConfigIter iter(&jsonxtc);
    unsigned r = _configure(xtc,iter);

    // append the config xtc info to the dgram
    memcpy((void*)xtc.next(),(const void*)jsonxtc.payload(),jsonxtc.sizeofPayload());
    xtc.alloc(jsonxtc.sizeofPayload());

    Py_DECREF(mybytes);
    Py_DECREF(json_bytes);
    delete[] buffer;

    return r;
}

unsigned BEBDetector::configureScan(const json& scan_keys,
                                    Xtc&        xtc)
{
    PyObject* pDict = _check(PyModule_GetDict(m_module));

    char func_name[64];
    sprintf(func_name,"%s_scan_keys",m_para->detType.c_str());
    PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)func_name));

    // returns new reference
    PyObject* mybytes = _check(PyObject_CallFunction(pFunc, "Os",
                                                     m_root, scan_keys.dump().c_str()));
     
    // returns new reference
    PyObject * json_bytes = _check(PyUnicode_AsASCIIString(mybytes));
    char* json = (char*)PyBytes_AsString(json_bytes);

    // convert json to xtc
    const int BUFSIZE = 1024*1024;
    char* buffer = new char[BUFSIZE];
    Xtc& jsonxtc = *new (buffer) Xtc(TypeId(TypeId::Parent, 0));  // Initialize xtc

    // append the scan config names to the xtc
    logging::debug("configureScan adding json %s [extent 0x%x]",json,jsonxtc.extent);
    Document* d = new Document();
    d->Parse(json);
    NamesId namesId(nodeId,UpdateNamesIndex);
    Value v;
    if (Pds::translateJson2XtcNames(d, &jsonxtc, m_namesLookup, namesId, v, m_para->detName.c_str(), m_para->detSegment) < 0)
        throw "**** Config json translation error\n";
    delete d; 
    if (jsonxtc.extent>BUFSIZE)
        throw "**** Config json output too large for buffer\n";

    // append the xtc info to the dgram
    memcpy((void*)xtc.next(),(const void*)jsonxtc.payload(),jsonxtc.sizeofPayload());
    xtc.alloc(jsonxtc.sizeofPayload());

    Py_DECREF(mybytes);
    Py_DECREF(json_bytes);
    delete[] buffer;

    return 0;
}

unsigned BEBDetector::stepScan(const json& stepInfo, Xtc& xtc)
{
    unsigned r=0;

    //  Call to python <det>_update function to apply to hw and generate full json changes
    PyObject* pDict = _check(PyModule_GetDict(m_module));

    char func_name[64];
    sprintf(func_name,"%s_update",m_para->detType.c_str());
    PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)func_name));

    // returns new reference
    PyObject* mybytes = _check(PyObject_CallFunction(pFunc, "Os",
                                                     m_root, stepInfo.dump().c_str()));
    
    // returns new reference
    PyObject * json_bytes = _check(PyUnicode_AsASCIIString(mybytes));
    char* json = (char*)PyBytes_AsString(json_bytes);
  
    logging::debug("stepScan json [%s]",json);

    // convert json to xtc
    const unsigned BUFSIZE = 1024*1024;
    char buffer[BUFSIZE];
    NamesId namesId(nodeId,UpdateNamesIndex);

    TypeId tid(TypeId::Parent, 0);

    Document* d = new Document();
    NamesLookup nl;
    Value v;

    d->Parse(json);
    Xtc* jsonxtc = new (buffer) Xtc(tid);
    {
        if (Pds::translateJson2XtcNames(d, jsonxtc, nl, namesId, v, 0, 0) < 0)
            throw "**** Config json translation error\n";
        if (jsonxtc->extent > BUFSIZE)
          throw "**** Config json output too large for buffer\n";
    }
    //  Overwrite
    jsonxtc = new (buffer) Xtc(tid);
    {
        if (Pds::translateJson2XtcData(d, jsonxtc, m_namesLookup, namesId, v) < 0)
            throw "**** Config json translation error\n";
        if (jsonxtc->extent > BUFSIZE)
          throw "**** Config json output too large for buffer\n";
    }

    logging::debug("Adding stepscan extent 0x%x",jsonxtc->extent);

    memcpy((void*)xtc.next(),(const void*)jsonxtc->payload(),jsonxtc->sizeofPayload());
    xtc.alloc(jsonxtc->sizeofPayload());
  
    Py_DECREF(mybytes);
    Py_DECREF(json_bytes);
  
    return r;
}

void BEBDetector::connect(const json& connect_json, const std::string& collectionId)
{
    logging::info("BEBDetector connect");
    m_connect_json = connect_json.dump();
    m_readoutGroup = connect_json["body"]["drp"][collectionId]["det_info"]["readout"];
}

void BEBDetector::event(XtcData::Dgram& dgram, PGPEvent* event)
{
    int lane = __builtin_ffs(event->mask) - 1;
    uint32_t dmaIndex = event->buffers[lane].index;
    unsigned data_size = event->buffers[lane].size;
    EvtBatcherIterator ebit = EvtBatcherIterator((EvtBatcherHeader*)m_pool->dmaBuffers[dmaIndex], data_size);
    EvtBatcherSubFrameTail* ebsft = ebit.next();
    unsigned nsubs = ebsft->tdest()+1;
    std::vector< XtcData::Array<uint8_t> > subframes(nsubs, XtcData::Array<uint8_t>(0, 0, 1) );

    do {
        subframes[ebsft->tdest()] = XtcData::Array<uint8_t>(ebsft->data(), &ebsft->size(), 1);
    } while ((ebsft=ebit.next()));

    _event(dgram.xtc, subframes);
}

void BEBDetector::shutdown()
{
    // returns borrowed reference
    PyObject* pDict = _check(PyModule_GetDict(m_module));

    char func_name[64];
    sprintf(func_name,"%s_unconfig",m_para->detType.c_str());
    // returns borrowed reference
    PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)func_name));

    // returns new reference
    PyObject* val = _check(PyObject_CallFunction(pFunc,"O",m_root));

    Py_DECREF(val);
}

}
