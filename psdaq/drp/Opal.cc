#include "Opal.hh"
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

static PyObject* check(PyObject* obj) {
    if (!obj) {
        PyErr_Print();
        throw "**** python error\n";
    }
    return obj;
}

static std::string _string_from_PyDict(PyObject* dict, const char* key)
{
  PyObject* item      = check(PyDict_GetItemString(dict,key));
  PyObject * item_str = check(PyUnicode_AsASCIIString(item));
  std::string s(PyBytes_AsString(item_str));
  Py_DECREF(item_str);
  return s;
}

class RawDef : public VarDef
{
public:
    enum index {
        image
    };
    RawDef()
    {
        Alg alg("raw", 2, 0, 0);
        NameVec.push_back({"image", Name::UINT16, 2});  // Does the data need to be reformatted?
    }
} rawDef;


Opal::Opal(Parameters* para, MemPool* pool) :
    Detector      (para, pool),
    m_evtNamesId  (-1, -1), // placeholder
    m_connect_json("")
{
    virtChan = 1;

    // returns new reference
    PyObject* pModule = check(PyImport_ImportModule("psalg.configdb.opal_config"));

    PyObject* pDict = check(PyModule_GetDict(pModule));
    {
      PyObject* pFunc = check(PyDict_GetItemString(pDict, (char*)"opal_init"));

      // returns new reference
#define MLOOKUP(m,name,dflt) (m.find(name)==m.end() ? dflt : m[name].c_str())
      const char* xpmpv = MLOOKUP(para->kwargs,"xpmpv",0);
      m_root = check(PyObject_CallFunction(pFunc,"s",xpmpv));
    }

    {
      PyObject* pFunc = check(PyDict_GetItemString(pDict, (char*)"opal_connect"));

      // returns new reference
      PyObject* mbytes = check(PyObject_CallFunction(pFunc,"O",m_root));

      m_paddr = PyLong_AsLong(PyDict_GetItemString(mbytes, "paddr"));
        
      if (!m_paddr) {
        const char msg[] = "XPM Remote link id register is zero\n";
        logging::error("%s", msg);
        throw msg;
      }

      unsigned modelnum = strtoul( _string_from_PyDict(mbytes,"model").c_str(), NULL, 10);
#define MODEL(num,rows,cols) case num: m_rows = rows; m_columns = cols; break
      switch(modelnum) {
        MODEL(1000,1024,1024);
        MODEL(1600,1200,1600);
        MODEL(2000,1080,1920);
        MODEL(4000,1752,2336);
        MODEL(8000,2472,3296);
#undef MODEL
      default:
        throw std::string("Opal camera model not recognized");
        break;
      }

      m_para->serNo = _string_from_PyDict(mbytes,"serno");

      Py_DECREF(mbytes);
    }

    //    Py_DECREF(pModule);
}

json Opal::connectionInfo()
{
    // Exclude connection info until cameralink-gateway timingTxLink is fixed
    logging::error("Returning NO XPM link; implementation incomplete");
    return json({});

    unsigned reg = m_paddr;
    int xpm  = (reg >> 20) & 0x0F;
    int port = (reg >>  0) & 0xFF;
    json info = {{"xpm_id", xpm}, {"xpm_port", port}};
    return info;
}

void Opal::_addJson(Xtc& xtc, NamesId& configNamesId, const std::string& config_alias) {

    // returns new reference
    PyObject* pModule = check(PyImport_ImportModule("psalg.configdb.opal_config"));

    PyObject* pDict = check(PyModule_GetDict(pModule));
    PyObject* pFunc = check(PyDict_GetItemString(pDict, (char*)"opal_config"));

    // returns new reference
    PyObject* mybytes = check(PyObject_CallFunction(pFunc, "Osssii", 
                                                    m_root, m_connect_json.c_str(), config_alias.c_str(), 
                                                    m_para->detName.c_str(), m_para->detSegment, m_readoutGroup));

    // returns new reference
    PyObject * json_bytes = check(PyUnicode_AsASCIIString(mybytes));
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

    Py_DECREF(pModule);
    Py_DECREF(mybytes);
    Py_DECREF(json_bytes);
}

void Opal::connect(const json& connect_json, const std::string& collectionId)
{
    logging::info("Opal connect");
    m_connect_json = connect_json.dump();

    int fd = open(m_para->device.c_str(), O_RDWR);
    if (fd < 0) {
        logging::error("Error opening %s", m_para->device.c_str());
        return;
    }

    m_readoutGroup = connect_json["body"]["drp"][collectionId]["det_info"]["readout"];

    close(fd);
}

unsigned Opal::configure(const std::string& config_alias, Xtc& xtc)
{
    m_evtNamesId = NamesId(nodeId, EventNamesIndex);
    // set up the names for the configuration data
    NamesId configNamesId(nodeId,ConfigNamesIndex);
    _addJson(xtc, configNamesId, config_alias);

    // set up the names for L1Accept data
    Alg alg("raw", 2, 0, 0);
    Names& eventNames = *new(xtc) Names(m_para->detName.c_str(), alg,
                                        m_para->detType.c_str(), m_para->serNo.c_str(), m_evtNamesId, m_para->detSegment);

    eventNames.add(xtc, rawDef);
    m_namesLookup[m_evtNamesId] = NameIndex(eventNames);
    return 0;
}

void Opal::event(XtcData::Dgram& dgram, PGPEvent* event)
{
    CreateData cd(dgram.xtc, m_namesLookup, m_evtNamesId);

    int lane = __builtin_ffs(event->mask) - 1;
    // there should be only one lane of data in the timing system
    uint32_t dmaIndex = event->buffers[lane].index;
    unsigned data_size = event->buffers[lane].size;
    EvtBatcherIterator ebit = EvtBatcherIterator((EvtBatcherHeader*)m_pool->dmaBuffers[dmaIndex], data_size);
    EvtBatcherSubFrameTail* ebsft;
    void*  image=0;
    size_t imageSize=0;
    void*  header=0;
    size_t headerSize=0;
    while ((ebsft=ebit.next())) {
        //printf("sft width %d size %d tdest %d\n",ebsft->width(),ebsft->size(),ebsft->tdest());
        if (ebsft->tdest()==0) {
            header = ebsft->data();
            headerSize = ebsft->size();
        }
        if (ebsft->tdest()==2) {
            image = ebsft->data();
            imageSize = ebsft->size();
        }
    }
    if (!header or !image) logging::critical("*** missing event header %p and/or image %p\n",header,image);
    if (headerSize!=32) logging::critical("*** incorrect header size %d\n",headerSize);

    unsigned shape[MaxRank];
    shape[0] = 1024;  // these depend upon the flavor of Opal (1000,2000,4000,8000)
    shape[1] = 1024;
    Array<uint8_t> arrayT = cd.allocate<uint8_t>(RawDef::image, shape);
    //  Do we need to reformat 12-bit into 16-bit?
    memcpy(arrayT.data(), image, imageSize);
}

}
