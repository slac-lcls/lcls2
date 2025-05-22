
#include "XpmDetector.hh"
#include "XpmInfo.hh"
#include "psalg/utils/SysLog.hh"
#include <unistd.h>
#include <netdb.h>
#include <arpa/inet.h>

using namespace XtcData;
using json = nlohmann::json;
using logging = psalg::SysLog;

namespace Drp {

static PyObject* _check(PyObject* obj) {
    if (!obj) {
        PyErr_Print();
        logging::critical("**** python error");
        abort();
    }
    return obj;
}

XpmDetector::XpmDetector(Parameters* para, MemPool* pool, unsigned len) :
    Detector(para, pool),
    m_length(len)
{
    _init();
}

void XpmDetector::_init()
{
    std::map<std::string,std::string>::iterator it = m_para->kwargs.find("timebase");
    const char* timebase = (it != m_para->kwargs.end()) ? it->second.data() : "186M";

    // returns new reference
    m_xmodule = _check(PyImport_ImportModule("psdaq.configdb.xpmdet_config"));

    PyObject* pDict = _check(PyModule_GetDict(m_xmodule));
    PyObject* pFunc = _check(PyDict_GetItemString(pDict, "xpmdet_init"));
    Py_DECREF(_check(PyObject_CallFunction(pFunc,"sisi",
                                           m_para->device.c_str(),
                                           m_para->laneMask,
                                           timebase,
                                           m_para->verbose)));
}

json XpmDetector::connectionInfo(const json& msg)
{
    PyObject* pDict  = _check(PyModule_GetDict(m_xmodule));
    PyObject* pFunc  = _check(PyDict_GetItemString(pDict, "xpmdet_connectionInfo"));
    PyObject* mbytes = _check(PyObject_CallFunction(pFunc,"s",msg.dump().c_str()));
    json result = xpmInfo(PyLong_AsLong(PyDict_GetItemString(mbytes, "paddr")));
    Py_DECREF(mbytes);
    return result;
}

// setup up device to receive data over pgp
void XpmDetector::connect(const json& connect_json, const std::string& collectionId)
{
    unsigned readoutGroup = connect_json["body"]["drp"][collectionId]["det_info"]["readout"];
    // FIXME make configureable
    std::map<std::string,std::string>::iterator it = m_para->kwargs.find("sim_length");
    if (it != m_para->kwargs.end())
        m_length = stoi(it->second);

    PyObject* pDict  = _check(PyModule_GetDict(m_xmodule));
    PyObject* pFunc  = _check(PyDict_GetItemString(pDict, "xpmdet_connect"));
    Py_DECREF(_check(PyObject_CallFunction(pFunc,"ii",readoutGroup,m_length)));
}

unsigned XpmDetector::configure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd)
{
    return 0;
}

void XpmDetector::shutdown()
{
    PyObject* pDict  = _check(PyModule_GetDict(m_xmodule));
    PyObject* pFunc  = _check(PyDict_GetItemString(pDict, "xpmdet_unconfig"));
    Py_DECREF(_check(PyObject_CallFunction(pFunc,"")));
}

void XpmDetector::connectionShutdown()
{
    PyObject* pDict  = _check(PyModule_GetDict(m_xmodule));
    PyObject* pFunc  = _check(PyDict_GetItemString(pDict, "xpmdet_connectionShutdown"));
    Py_DECREF(_check(PyObject_CallFunction(pFunc,"")));
}
}
