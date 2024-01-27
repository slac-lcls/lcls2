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
        logging::critical("**** python error");
        abort();
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
    m_module      (0),
    m_configScanner (0),
    m_debatch     (false)
{
    virtChan = 1;
}

BEBDetector::~BEBDetector()
{
    delete m_configScanner;
    if (m_module)  Py_DECREF(m_module);
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
      const char* timebase = MLOOKUP(m_para->kwargs,"timebase","186M");

      m_root = _check(PyObject_CallFunction(pFunc,"ssissi",
                                            arg,
                                            m_para->device.c_str(),
                                            m_para->laneMask,
                                            xpmpv,
                                            timebase,
                                            m_para->verbose));

      // check if m_root has "virtChan" member and set accordingly
      if (m_root) {
          PyObject* o_virtChan = PyDict_GetItemString(m_root,"virtChan");
          if (o_virtChan)
              virtChan = PyLong_AsLong(o_virtChan);
      }
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

json BEBDetector::connectionInfo(const json& msg)
{
    std::string alloc_json = msg.dump();
    char func_name[64];
    PyObject* pDict = _check(PyModule_GetDict(m_module));
    {
      sprintf(func_name,"%s_connectionInfo",m_para->detType.c_str());
      PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)func_name));

      // returns new reference
      PyObject* mbytes = _check(PyObject_CallFunction(pFunc,"Os",m_root,alloc_json.c_str()));

      m_paddr = PyLong_AsLong(PyDict_GetItemString(mbytes, "paddr"));
      printf("*** BebDetector: paddr is %08x = %u\n", m_paddr, m_paddr);

      // there is currently a failure mode where the register reads
      // back as zero or 0xffffffff (incorrectly). This is not the best
      // longterm fix, but throw here to highlight the problem. the
      // difficulty is that Matt says this register has to work
      // so that an automated software solution would know which
      // xpm TxLink's to reset (a chicken-and-egg problem) - cpo
      // Also, register is corrupted when port number > 15 - Ric
      if (!m_paddr || m_paddr==0xffffffff || (m_paddr & 0xff) > 15) {
          logging::critical("XPM Remote link id register illegal value: 0x%x. Try XPM TxLink reset.",m_paddr);
          abort();
      }

      _connectionInfo(mbytes);

      Py_DECREF(mbytes);
    }

    unsigned reg = m_paddr;
    int xpm  = (reg >> 20) & 0x0F;
    int port = (reg >>  0) & 0xFF;
    json info = {{"xpm_id", xpm}, {"xpm_port", port}};
    printf("*** BEBDet %d %d %x\n",xpm,port,m_paddr);
    return info;
}

void BEBDetector::connect(const json& connect_json, const std::string& collectionId)
{
    logging::info("BEBDetector connect");
    m_connect_json = connect_json.dump();
    m_readoutGroup = connect_json["body"]["drp"][collectionId]["det_info"]["readout"];
}

unsigned BEBDetector::configure(const std::string& config_alias,
                                Xtc&               xtc,
                                const void*        bufEnd)
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
    const void* end = buffer + m_para->maxTrSize;
    Xtc& jsonxtc = *new (buffer, end) Xtc(TypeId(TypeId::Parent, 0));

    logging::debug("PyList_Check");
    if (PyList_Check(mybytes)) {
        logging::debug("PyList_Check true");
        for(unsigned seg=0; seg<PyList_Size(mybytes); seg++) {
            logging::debug("seg %d",seg);
            PyObject* item = PyList_GetItem(mybytes,seg);
            logging::debug("item %p",item);
            NamesId namesId(nodeId,ConfigNamesIndex+seg);
            if (Pds::translateJson2Xtc( item, jsonxtc, end, namesId ))
                return -1;
        }
    }
    else if ( Pds::translateJson2Xtc( mybytes, jsonxtc, end, NamesId(nodeId,ConfigNamesIndex) ) )
        return -1;

    if (jsonxtc.extent>m_para->maxTrSize)
        throw "**** Config json output too large for buffer\n";

    XtcData::ConfigIter iter(&jsonxtc, end);
    unsigned r = _configure(xtc,bufEnd,iter);

    // append the config xtc info to the dgram
    auto payload = xtc.alloc(jsonxtc.sizeofPayload(), bufEnd);
    memcpy(payload,(const void*)jsonxtc.payload(),jsonxtc.sizeofPayload());

    Py_DECREF(mybytes);
    delete[] buffer;

    return r;
}

unsigned BEBDetector::configureScan(const json& scan_keys,
                                    Xtc&        xtc,
                                    const void* bufEnd)
{
    NamesId namesId(nodeId,UpdateNamesIndex);
    return m_configScanner->configure(scan_keys,xtc,bufEnd,namesId,m_namesLookup);
}

unsigned BEBDetector::stepScan(const json& stepInfo, Xtc& xtc, const void* bufEnd)
{
    NamesId namesId(nodeId,UpdateNamesIndex);
    return m_configScanner->step(stepInfo,xtc,bufEnd,namesId,m_namesLookup);
}

Pds::TimingHeader* BEBDetector::getTimingHeader(uint32_t index) const
{
    EvtBatcherHeader* ebh = static_cast<EvtBatcherHeader*>(m_pool->dmaBuffers[index]);
    if (m_debatch) ebh = reinterpret_cast<EvtBatcherHeader*>(ebh->next());
    return static_cast<Pds::TimingHeader*>(ebh->next());
}

std::vector< XtcData::Array<uint8_t> > BEBDetector::_subframes(void* buffer, unsigned length)
{
    EvtBatcherIterator ebit = EvtBatcherIterator((EvtBatcherHeader*)buffer, length);
    EvtBatcherSubFrameTail* ebsft = ebit.next();
    unsigned nsubs = ebsft->tdest()+1;
    std::vector< XtcData::Array<uint8_t> > subframes(nsubs, XtcData::Array<uint8_t>(0, 0, 1) );
    do {
        logging::debug("Deb::event: array[%d] sz[%d]\n",ebsft->tdest(),ebsft->size());
        subframes[ebsft->tdest()] = XtcData::Array<uint8_t>(ebsft->data(), &ebsft->size(), 1);
    } while ((ebsft=ebit.next()));
    return subframes;
}

void BEBDetector::event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event)
{
    int lane = __builtin_ffs(event->mask) - 1;
    uint32_t dmaIndex = event->buffers[lane].index;
    unsigned data_size = event->buffers[lane].size;

    std::vector< XtcData::Array<uint8_t> > subframes = _subframes(m_pool->dmaBuffers[dmaIndex], data_size);
    if (m_debatch)
        subframes = _subframes(subframes[2].data(), subframes[2].shape()[0]);
    _event(dgram.xtc, bufEnd, subframes);
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
    PyObject* val = PyObject_CallFunction(pFunc,"O",m_root);

    if (val)
        Py_DECREF(val);
    else
        PyErr_Print();

}

}
