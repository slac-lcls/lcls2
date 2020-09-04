#include "PythonConfigScanner.hh"

#include "psdaq/service/Json2Xtc.hh"
#include "psalg/utils/SysLog.hh"
#include "drp.hh"
#include <Python.h>

using namespace XtcData;
using namespace rapidjson;
using json = nlohmann::json;
using namespace Drp;
using logging = psalg::SysLog;

static PyObject* _check(PyObject* obj) {
    if (!obj) {
        PyErr_Print();
        throw "**** python error\n";
    }
    return obj;
}

PythonConfigScanner::PythonConfigScanner(const Parameters& para,
                                         PyObject&   module) :
    m_para  (para),
    m_module(module)
{
}

PythonConfigScanner::~PythonConfigScanner()
{
}

unsigned PythonConfigScanner::configure(const json&  scan_keys, 
                                        Xtc&         xtc,
                                        NamesId&     namesId,
                                        NamesLookup& namesLookup)
{
    PyObject* pDict = _check(PyModule_GetDict(&m_module));

    char func_name[64];
    sprintf(func_name,"%s_scan_keys",m_para.detType.c_str());
    PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)func_name));

    // returns new reference
    PyObject* mybytes = _check(PyObject_CallFunction(pFunc, "s",
                                                     scan_keys.dump().c_str()));
     
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
    Value v;
    if (Pds::translateJson2XtcNames(d, &jsonxtc, namesLookup, namesId, v, m_para.detName.c_str(), m_para.detSegment) < 0)
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

unsigned PythonConfigScanner::step(const json&  stepInfo, 
                                   Xtc&         xtc,
                                   NamesId&     namesId,
                                   NamesLookup& namesLookup)
{
    unsigned r=0;

    //  Call to python <det>_update function to apply to hw and generate full json changes
    PyObject* pDict = _check(PyModule_GetDict(&m_module));

    char func_name[64];
    sprintf(func_name,"%s_update",m_para.detType.c_str());
    PyObject* pFunc = _check(PyDict_GetItemString(pDict, (char*)func_name));

    // returns new reference
    PyObject* mybytes = _check(PyObject_CallFunction(pFunc, "s",
                                                     stepInfo.dump().c_str()));
    
    // returns new reference
    PyObject * json_bytes = _check(PyUnicode_AsASCIIString(mybytes));
    char* json = (char*)PyBytes_AsString(json_bytes);
  
    logging::debug("stepScan json [%s]",json);

    // convert json to xtc
    const unsigned BUFSIZE = 1024*1024;
    char buffer[BUFSIZE];

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
        if (Pds::translateJson2XtcData(d, jsonxtc, namesLookup, namesId, v) < 0)
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
