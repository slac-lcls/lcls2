#include "BEBDetector.hh"
#include "PythonConfigScanner.hh"
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
        throw "**** python error";
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
    delete m_configScanner;
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
      m_root = _check(PyObject_CallFunction(pFunc,"ssis",
                                            arg,
                                            m_para->device.c_str(),
                                            m_para->laneMask,
                                            xpmpv));
    }

    {
      sprintf(func_name,"%s_connect",m_para->detType.c_str());
      PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)func_name));

      // returns new reference
      PyObject* mbytes = _check(PyObject_CallFunction(pFunc,"O",m_root));

      m_paddr = PyLong_AsLong(PyDict_GetItemString(mbytes, "paddr"));

      if (!m_paddr) {
        throw "XPM Remote link id register is zero";
      }

      _connect(mbytes);

      Py_DECREF(mbytes);
    }

    m_configScanner = new PythonConfigScanner(*m_para,*m_module);
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

    char* buffer = new char[m_para->maxTrSize];
    Xtc& jsonxtc = *new (buffer) Xtc(TypeId(TypeId::Parent, 0));

    if (PyList_Check(mybytes)) {
        for(unsigned seg=0; seg<PyList_Size(mybytes); seg++) {
            PyObject* item = PyList_GetItem(mybytes,seg);
            NamesId namesId(nodeId,ConfigNamesIndex+seg);
            if (Pds::translateJson2Xtc( item, jsonxtc, namesId ))
                return -1;
        }
    }
    else if ( Pds::translateJson2Xtc( mybytes, jsonxtc, NamesId(nodeId,ConfigNamesIndex) ) )
        return -1;

    if (jsonxtc.extent>m_para->maxTrSize)
        throw "**** Config json output too large for buffer\n";

    XtcData::ConfigIter iter(&jsonxtc);
    unsigned r = _configure(xtc,iter);
        
    // append the config xtc info to the dgram
    memcpy((void*)xtc.next(),(const void*)jsonxtc.payload(),jsonxtc.sizeofPayload());
    xtc.alloc(jsonxtc.sizeofPayload());

    Py_DECREF(mybytes);
    delete[] buffer;

    return r;
}

unsigned BEBDetector::configureScan(const json& scan_keys,
                                    Xtc&        xtc)
{
    NamesId namesId(nodeId,UpdateNamesIndex);
    return m_configScanner->configure(scan_keys,xtc,namesId,m_namesLookup);
}

unsigned BEBDetector::stepScan(const json& stepInfo, Xtc& xtc)
{
    NamesId namesId(nodeId,UpdateNamesIndex);
    return m_configScanner->step(stepInfo,xtc,namesId,m_namesLookup);
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
