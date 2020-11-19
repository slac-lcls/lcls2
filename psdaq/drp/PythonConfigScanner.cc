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

static const int BUFSIZE = 4*1024*1024;

//
//  On Configure, save the Names for the keys that will change
//
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
     
    // convert json to xtc
    char* buffer = new char[BUFSIZE];
    Xtc& jsonxtc = *new (buffer) Xtc(TypeId(TypeId::Parent, 0));  // Initialize xtc

    //    Pds::setJsonDebug();

    bool isList = PyList_Check(mybytes);
    unsigned nseg = isList ? PyList_Size(mybytes) : 1;

    unsigned nodeId   = namesId.nodeId();
    unsigned namesIdx = namesId.namesId();
    for(unsigned seg=0; seg<nseg; seg++) {
        PyObject* item = isList ? PyList_GetItem(mybytes,seg) : mybytes;
        NamesId nId(nodeId,namesIdx+seg);

        PyObject * json_bytes = PyUnicode_AsASCIIString(item);
        char* json = (char*)PyBytes_AsString(json_bytes);

        Document *d = new Document();
        d->Parse(json);

        //  Extract detname, segment from document detName field
        const char* detname=0;
        unsigned segment = 0;
        std::string sdetName((*d)["detName:RO"].GetString());
        {
            size_t pos = sdetName.rfind('_');
            if (pos==std::string::npos) {
                logging::error("No segment number in config json");
                break;
            }
            sscanf(sdetName.c_str()+pos+1,"%u",&segment);
            detname = sdetName.substr(0,pos).c_str();
        }

        Value jsonv;
        if (Pds::translateJson2XtcNames(d, &jsonxtc, namesLookup, nId, jsonv, detname, segment) < 0)
            return -1;
        
        delete d;
        Py_DECREF(json_bytes);
    }

    if (jsonxtc.extent>BUFSIZE)
        throw "**** Config json output too large for buffer\n";

    // append the xtc info to the dgram
    memcpy((void*)xtc.next(),(const void*)jsonxtc.payload(),jsonxtc.sizeofPayload());
    xtc.alloc(jsonxtc.sizeofPayload());

    Py_DECREF(mybytes);
    delete[] buffer;

    return 0;
}

static int _translate( PyObject* item, Xtc* xtc, NamesLookup& namesLookup, NamesId namesID) 
{
    int result = -1;

    // returns new reference
    PyObject * json_bytes = PyUnicode_AsASCIIString(item);
    char* json = (char*)PyBytes_AsString(json_bytes);

    NamesLookup nl;
    Value jsonv;

    Document *d = new Document();
    d->Parse(json);

    while(1) {
        const char* detname;
        unsigned    segment = 0;
        if (!d->HasMember("detName:RO")) {
            logging::info("No detName member in config json");
            break;
        }

        //  Extract detname, segment from document detName field
        std::string sdetName((*d)["detName:RO"].GetString());
        {
            size_t pos = sdetName.rfind('_');
            if (pos==std::string::npos) {
                logging::info("No segment number in config json");
                break;
            }
            sscanf(sdetName.c_str()+pos+1,"%u",&segment);
            detname = sdetName.substr(0,pos).c_str();
        }

        unsigned extent = xtc->extent;
        logging::info("update names");
        if (Pds::translateJson2XtcNames(d, xtc, nl, namesID, jsonv, detname, segment) < 0)
            break;
    
        xtc->extent = extent;
        logging::info("update data");
        if (Pds::translateJson2XtcData (d, xtc, namesLookup, namesID, jsonv) < 0)
            break;

        result = 0;
        break;
    }

    delete d;
    Py_DECREF(json_bytes);

    return result;
}

//
//  On BeginStep, save the updated configuration data (matching the keys stored on Configure)
//
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
  
    logging::info("stepScan json [%s]",json);

    // convert json to xtc
    const unsigned BUFSIZE = 1024*1024;
    char buffer[BUFSIZE];
    Xtc* jsonxtc = new (buffer) Xtc(TypeId(TypeId::Parent, 0));

    if (PyList_Check(mybytes)) {
        for(unsigned seg=0; seg<PyList_Size(mybytes); seg++) {
            logging::info("scan update seg %d",seg);
            PyObject* item = PyList_GetItem(mybytes,seg);
            NamesId nId(namesId.nodeId(),namesId.namesId()+seg);
            if (_translate( item, jsonxtc, namesLookup, nId ))
                return -1;
        }
    }
    else if ( _translate( mybytes, jsonxtc, namesLookup, namesId) )
        return -1;

    if (jsonxtc->extent>BUFSIZE)
        throw "**** Config json output too large for buffer\n";

    logging::info("Adding stepscan extent 0x%x",jsonxtc->extent);

    memcpy((void*)xtc.next(),(const void*)jsonxtc->payload(),jsonxtc->sizeofPayload());
    xtc.alloc(jsonxtc->sizeofPayload());
  
    Py_DECREF(mybytes);
  
    return r;
}
