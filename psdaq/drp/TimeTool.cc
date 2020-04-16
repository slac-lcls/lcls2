#include "TimeTool.hh"
#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psdaq/service/Json2Xtc.hh"
#include "rapidjson/document.h"
#include "xtcdata/xtc/XtcIterator.hh"
#include "psalg/utils/SysLog.hh"
#include "AxisDriver.h"

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

class TTDef : public VarDef
{
public:
    enum index {
        image
    };
    TTDef()
    {
        Alg alg("tt", 2, 0, 0);
        NameVec.push_back({"image", Name::UINT8, 1});
    }
} TTDef;

TimeTool::TimeTool(Parameters* para, MemPool* pool) :
    Detector(para, pool),
    m_evtNamesId(-1, -1), // placeholder
    m_connect_json(""),
    m_paddr       (_getPaddr())
{
    virtChan = 1;
}

static void check(PyObject* obj) {
    if (!obj) {
        PyErr_Print();
        throw "**** python error\n";
    }
}

unsigned TimeTool::_getPaddr() {
    // returns new reference
    PyObject* pModule = PyImport_ImportModule("psalg.config.tt_config");
    check(pModule);
    // returns borrowed reference
    PyObject* pDict = PyModule_GetDict(pModule);
    check(pDict);
    // returns borrowed reference
    PyObject* pFunc = PyDict_GetItemString(pDict, (char*)"tt_connect");
    check(pFunc);
    // returns new reference
    PyObject* mybytes = PyObject_CallFunction(pFunc, "i", 0);
    check(mybytes);
    // returns new reference
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
}

json TimeTool::connectionInfo()
{
    unsigned reg = m_paddr;
    if (!reg) {
        const char msg[] = "XPM Remote link id register is zero\n";
        logging::error("%s", msg);
        throw msg;
    }
    int xpm  = (reg >> 20) & 0x0F;
    int port = (reg >>  0) & 0xFF;
    json info = {{"xpm_id", xpm}, {"xpm_port", port}};
    return info;
}

void TimeTool::_addJson(Xtc& xtc, NamesId& configNamesId, const std::string& config_alias) {

    // returns new reference
    PyObject* pModule = PyImport_ImportModule("psalg.configdb.tt_config");
    check(pModule);
    // returns borrowed reference
    PyObject* pDict = PyModule_GetDict(pModule);
    check(pDict);
    // returns borrowed reference
    PyObject* pFunc = PyDict_GetItemString(pDict, (char*)"tt_config");
    check(pFunc);
    // returns new reference
    PyObject* mybytes = PyObject_CallFunction(pFunc, "sssii", 
                                              m_connect_json.c_str(), config_alias.c_str(), 
                                              m_para->detName.c_str(), m_para->detSegment, m_readoutGroup);
    check(mybytes);

    // returns new reference
    PyObject * json_bytes = PyUnicode_AsASCIIString(mybytes);
    check(json_bytes);
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

void TimeTool::connect(const json& connect_json, const std::string& collectionId)
{
    logging::info("TimeTool connect");
    m_connect_json = connect_json.dump();

    int fd = open(m_para->device.c_str(), O_RDWR);
    if (fd < 0) {
        logging::error("Error opening %s", m_para->device.c_str());
        return;
    }

    m_readoutGroup = connect_json["body"]["drp"][collectionId]["det_info"]["readout"];

    // the address comes from pyrogue rootDevice.saveAddressMap('fname')
    // perhaps ideally we would call detector-specific rogue python
    // or use the new rogue version 5 c++ interface.
    dmaWriteRegister(fd, 0x940104, m_readoutGroup);

    close(fd);
}

unsigned TimeTool::configure(const std::string& config_alias, Xtc& xtc)
{
    m_evtNamesId = NamesId(nodeId, EventNamesIndex);
    // set up the names for the configuration data
    NamesId configNamesId(nodeId,ConfigNamesIndex);
    _addJson(xtc, configNamesId, config_alias);

    // set up the names for L1Accept data
    Alg alg("raw", 2, 0, 0);
    Names& eventNames = *new(xtc) Names(m_para->detName.c_str(), alg,
                                        m_para->detType.c_str(), m_para->serNo.c_str(), m_evtNamesId, m_para->detSegment);
    eventNames.add(xtc, TTDef);
    m_namesLookup[m_evtNamesId] = NameIndex(eventNames);
    return 0;
}

void TimeTool::event(XtcData::Dgram& dgram, PGPEvent* event)
{
    CreateData tt(dgram.xtc, m_namesLookup, m_evtNamesId);

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
    if (!header or !image) logging::critical("*** missing timetool header %p and/or image %p\n",header,image);
    if (headerSize!=32) logging::critical("*** incorrect header size %d\n",headerSize);

    unsigned shape[MaxRank];
    shape[0] = imageSize;
    Array<uint8_t> arrayT = tt.allocate<uint8_t>(TTDef::image, shape);
    memcpy(arrayT.data(), image, imageSize);
}

}
