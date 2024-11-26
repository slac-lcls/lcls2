
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

XpmDetector::XpmDetector(Parameters* para, MemPool* pool) :
    Detector(para, pool)
{
    _init();
}

void XpmDetector::_init()
{
    std::map<std::string,std::string>::iterator it = m_para->kwargs.find("timebase");
    const char* timebase = (it != m_para->kwargs.end()) ? it->second.data() : "186M";

    // returns new reference
    m_module = _check(PyImport_ImportModule("psdaq.configdb.xpmdet_config"));

    PyObject* pDict = _check(PyModule_GetDict(m_module));
    PyObject* pFunc = _check(PyDict_GetItemString(pDict, "xpmdet_init"));
    _check(PyObject_CallFunction(pFunc,"sisi",
                                 m_para->device.c_str(),
                                 m_para->laneMask,
                                 timebase,
                                 m_para->verbose));
}

json XpmDetector::connectionInfo(const json& msg)
{
    PyObject* pDict  = _check(PyModule_GetDict(m_module));
    PyObject* pFunc  = _check(PyDict_GetItemString(pDict, "xpmdet_connectionInfo"));
    PyObject* mbytes = _check(PyObject_CallFunction(pFunc,"s",msg.dump().c_str()));
    return xpmInfo(PyLong_AsLong(PyDict_GetItemString(mbytes, "paddr")));
}

// setup up device to receive data over pgp
void XpmDetector::connect(const json& connect_json, const std::string& collectionId)
{
    unsigned readoutGroup = connect_json["body"]["drp"][collectionId]["det_info"]["readout"];
    // FIXME make configureable
    unsigned length = 100;
    std::map<std::string,std::string>::iterator it = m_para->kwargs.find("sim_length");
    if (it != m_para->kwargs.end())
        length = stoi(it->second);

    PyObject* pDict  = _check(PyModule_GetDict(m_module));
    PyObject* pFunc  = _check(PyDict_GetItemString(pDict, "xpmdet_connect"));
    _check(PyObject_CallFunction(pFunc,"ii",readoutGroup,length));
}

unsigned XpmDetector::configure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd)
{
    return 0;
}

void XpmDetector::shutdown()
{
    PyObject* pDict  = _check(PyModule_GetDict(m_module));
    PyObject* pFunc  = _check(PyDict_GetItemString(pDict, "xpmdet_unconfig"));
    _check(PyObject_CallFunction(pFunc,""));
}
}
