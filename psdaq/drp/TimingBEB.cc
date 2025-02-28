#include "TimingBEB.hh"
#include "TimingDef.hh"
#include "psdaq/service/Semaphore.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "psalg/utils/SysLog.hh"
#include "psalg/detector/UtilsConfig.hh"

#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <fcntl.h>
#include <stdint.h>

using namespace XtcData;
using logging = psalg::SysLog;
using json = nlohmann::json;

static Drp::TimingDef TSDef;

using Drp::TimingBEB;

TimingBEB::TimingBEB(Parameters* para, MemPool* pool) :
    BEBDetector   (para, pool)
{
    _init(para->detName.c_str());  // an argument is required here
    _init_feb();
}

TimingBEB::~TimingBEB()
{
}

void TimingBEB::_connectionInfo(PyObject* mbytes)
{
    m_para->serNo = _string_from_PyDict(mbytes,"serno");
}

void TimingBEB::connect(const json&        connect_json,
                        const std::string& collectionId)
{
    logging::info("TimingBEB connect");
    BEBDetector::connect(connect_json, collectionId);

    // returns new reference
    PyObject* pModule = PyImport_ImportModule("psdaq.configdb.ts_connect");
    _check(pModule);
    // returns borrowed reference
    PyObject* pDict = PyModule_GetDict(pModule);
    _check(pDict);
    // returns borrowed reference
    PyObject* pFunc = PyDict_GetItemString(pDict, (char*)"ts_connect");
    _check(pFunc);
    // returns new reference
    PyObject* mybytes = PyObject_CallFunction(pFunc,"s",m_connect_json.c_str());
    _check(mybytes);

    Py_DECREF(pModule);
    Py_DECREF(mybytes);
}

unsigned TimingBEB::_configure(XtcData::Xtc& xtc, const void* bufEnd, XtcData::ConfigIter& configo)
{
    // set up the names for L1Accept data
    m_evtNamesId = NamesId(nodeId, EventNamesIndex);
    Alg alg("raw", 2, 1, 0);
    Names& eventNames = *new(xtc, bufEnd) Names(bufEnd, m_para->detName.c_str(), alg,
                                                "ts", m_para->serNo.c_str(), m_evtNamesId, m_para->detSegment);
    eventNames.add(xtc, bufEnd, TSDef);
    m_namesLookup[m_evtNamesId] = NameIndex(eventNames);
    return 0;
}

void TimingBEB::_event(XtcData::Xtc& xtc, const void* bufEnd, std::vector< XtcData::Array<uint8_t> >& subframes)
{
    TSDef.createDataETM(xtc, bufEnd, m_namesLookup, m_evtNamesId, subframes[0].data(), subframes[subframes.size()-1].data());
}

bool     TimingBEB::scanEnabled()
{
    return true;
}

void     TimingBEB::shutdown()
{
}

