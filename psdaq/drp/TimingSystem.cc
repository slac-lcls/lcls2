#include "TimingSystem.hh"
#include "PythonConfigScanner.hh"
#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psdaq/service/Json2Xtc.hh"
#include "rapidjson/document.h"
#include "xtcdata/xtc/XtcIterator.hh"
#include "AxisDriver.h"
#include "DataDriver.h"

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

static PyObject* check(PyObject* obj) {
    if (!obj) {
        PyErr_Print();
        throw "**** python error\n";
    }
    return obj;
}

namespace Drp {

class TSDef : public VarDef
{
public:
    enum index {
      pulseId,
      timeStamp,
      fixedRates,
      acRates,
      timeSlot,
      timeSlotPhase,
      ebeamPresent,
      ebeamDestn,
      ebeamCharge,
      ebeamEnergy,
      xWavelength,
      dmod5,
      mpsLimits,
      mpsPowerClass,
      sequenceValues,
    };
    TSDef()
    {
        NameVec.push_back({"pulseId"       , Name::UINT64, 0});
        NameVec.push_back({"timeStamp"     , Name::UINT64, 0});
        NameVec.push_back({"fixedRates"    , Name::UINT8 , 1});
        NameVec.push_back({"acRates"       , Name::UINT8 , 1});
        NameVec.push_back({"timeSlot"      , Name::UINT8 , 0});
        NameVec.push_back({"timeSlotPhase" , Name::UINT16, 0});
        NameVec.push_back({"ebeamPresent"  , Name::UINT8 , 0});
        NameVec.push_back({"ebeamDestn"    , Name::UINT8 , 0});
        NameVec.push_back({"ebeamCharge"   , Name::UINT16, 0});
        NameVec.push_back({"ebeamEnergy"   , Name::UINT16, 1});
        NameVec.push_back({"xWavelength"   , Name::UINT16, 1});
        NameVec.push_back({"dmod5"         , Name::UINT16, 0});
        NameVec.push_back({"mpsLimits"     , Name::UINT8 , 1});
        NameVec.push_back({"mpsPowerClass" , Name::UINT8 , 1});
        NameVec.push_back({"sequenceValues", Name::UINT16, 1});
    }
} TSDef;

TimingSystem::TimingSystem(Parameters* para, MemPool* pool) :
    XpmDetector   (para, pool),
    m_connect_json(""),
    m_module      (0)
{
    char module_name[64];
    sprintf(module_name,"psdaq.configdb.%s_config",para->detType.c_str());

    // returns new reference
    m_module = check(PyImport_ImportModule(module_name));

    m_configScanner = new PythonConfigScanner(*m_para,*m_module);
}

TimingSystem::~TimingSystem()
{
    delete m_configScanner;
    Py_DECREF(m_module);
}

void TimingSystem::_addJson(Xtc& xtc, NamesId& configNamesId, const std::string& config_alias) {

    // returns borrowed reference
    PyObject* pDict = check(PyModule_GetDict(m_module));
    check(pDict);
    // returns borrowed reference
    PyObject* pFunc = PyDict_GetItemString(pDict, (char*)"ts_config");
    check(pFunc);
    // need to get the dbase connection info via collection
    // returns new reference
    PyObject* mybytes = PyObject_CallFunction(pFunc,"sssi",m_connect_json.c_str(), config_alias.c_str(),
                                              m_para->detName.c_str(), m_para->detSegment);
    check(mybytes);
    // returns new reference
    PyObject * json_bytes = PyUnicode_AsASCIIString(mybytes);
    check(json_bytes);
    char* json = (char*)PyBytes_AsString(json_bytes);

    // convert to json to xtc
    unsigned len = Pds::translateJson2Xtc(json, config_buf, configNamesId, m_para->detName.c_str(), m_para->detSegment);
    if (len>BUFSIZE) {
        throw "**** Config json output too large for buffer\n";
    }
    if (len <= 0) {
        throw "**** Config json translation error\n";
    }

    // append the config xtc info to the dgram
    Xtc& jsonxtc = *(Xtc*)config_buf;
    memcpy((void*)xtc.next(),(const void*)jsonxtc.payload(),jsonxtc.sizeofPayload());
    xtc.alloc(jsonxtc.sizeofPayload());

    Py_DECREF(mybytes);
    Py_DECREF(json_bytes);

}

// TODO: put timeout value in connect and attach (conceptually like Collection.cc CollectionApp::handlePlat)

void TimingSystem::connect(const json& connect_json, const std::string& collectionId)
{
    XpmDetector::connect(connect_json, collectionId);

    int fd = m_pool->fd();
    int links = m_para->laneMask;

    AxiVersion vsn;
    axiVersionGet(fd, &vsn);
    if (vsn.userValues[2]) // Second PCIe interface has lanes shifted by 4
       links <<= 4;

    for(unsigned i=0, l=links; l; i++) {
        if (l&(1<<i)) {
          dmaWriteRegister(fd, 0x00a00000+4*(i&3), (1<<30));  // clear
          dmaWriteRegister(fd, 0x00a00000+4*(i&3), (1<<31));  // enable, zero-out length
          l &= ~(1<<i);
        }
    }

    // returns new reference
    PyObject* pModule = PyImport_ImportModule("psdaq.configdb.ts_connect");
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
    if (XpmDetector::configure(config_alias, xtc))
        return 1;

    m_evtNamesId = NamesId(nodeId, EventNamesIndex);
    // set up the names for the configuration data
    NamesId configNamesId(nodeId,ConfigNamesIndex);
    _addJson(xtc, configNamesId, config_alias);

    // set up the names for L1Accept data
    Alg alg("raw", 2, 0, 0);
    Names& eventNames = *new(xtc) Names(m_para->detName.c_str(), alg,
                                        m_para->detType.c_str(), m_para->serNo.c_str(), m_evtNamesId, m_para->detSegment);
    eventNames.add(xtc, TSDef);
    m_namesLookup[m_evtNamesId] = NameIndex(eventNames);
    return 0;
}

unsigned TimingSystem::configureScan(const nlohmann::json& scan_keys, XtcData::Xtc& xtc) 
{
    NamesId namesId(nodeId,UpdateNamesIndex);
    return m_configScanner->configure(scan_keys, xtc, namesId, m_namesLookup);
}

unsigned TimingSystem::beginstep(XtcData::Xtc& xtc, const json& stepInfo) {
    std::cout << "*** stepInfo: " << stepInfo.dump() << std::endl;
    return 0;
}

unsigned TimingSystem::stepScan(const json& stepInfo, Xtc& xtc)
{
    NamesId namesId(nodeId,UpdateNamesIndex);
    return m_configScanner->step(stepInfo, xtc, namesId, m_namesLookup);
}


// returning true here causes this detector to record scanInfo data
bool TimingSystem::scanEnabled() {
    return true;
}

void TimingSystem::event(XtcData::Dgram& dgram, PGPEvent* event)
{
    DescribedData ts(dgram.xtc, m_namesLookup, m_evtNamesId);

    int lane = __builtin_ffs(event->mask) - 1;
    // there should be only one lane of data in the timing system
    uint32_t dmaIndex  = event->buffers[lane].index;
    //    unsigned data_size = event->buffers[lane].size - sizeof(Pds::TimingHeader);
    // DMA is padded to workaround firmware problem; only copy relevant part.
    unsigned data_size = 968/8;

    memcpy(ts.data(), (uint8_t*)m_pool->dmaBuffers[dmaIndex] + sizeof(Pds::TimingHeader), data_size);
    ts.set_data_length(data_size);

    ts.set_array_shape(TSDef::fixedRates    ,(const unsigned[MaxRank]){10});
    ts.set_array_shape(TSDef::acRates       ,(const unsigned[MaxRank]){ 6});
    ts.set_array_shape(TSDef::ebeamEnergy   ,(const unsigned[MaxRank]){ 4});
    ts.set_array_shape(TSDef::xWavelength   ,(const unsigned[MaxRank]){ 2});
    ts.set_array_shape(TSDef::mpsLimits     ,(const unsigned[MaxRank]){16});
    ts.set_array_shape(TSDef::mpsPowerClass ,(const unsigned[MaxRank]){16});
    ts.set_array_shape(TSDef::sequenceValues,(const unsigned[MaxRank]){18});
}

}
