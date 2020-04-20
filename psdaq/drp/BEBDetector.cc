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
    m_connect_json("")
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

json BEBDetector::connectionInfo()
{
    unsigned reg = m_paddr;
    int xpm  = (reg >> 20) & 0x0F;
    int port = (reg >>  0) & 0xFF;
    json info = {{"xpm_id", xpm}, {"xpm_port", port}};
    return info;
}

unsigned BEBDetector::configure(const std::string& config_alias, Xtc& xtc)
{
    NamesId configNamesId(nodeId,ConfigNamesIndex);

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

    // convert to json to xtc
    const unsigned BUFSIZE = 1024*1024;
    char buffer[BUFSIZE];
    unsigned len = Pds::translateJson2Xtc(json, buffer, configNamesId, m_para->detName.c_str(), m_para->detSegment);
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

    Py_DECREF(mybytes);
    Py_DECREF(json_bytes);

    return _configure(xtc);
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
