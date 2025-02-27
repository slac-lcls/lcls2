/**
 **  Class to support scanning of python-implemented detector configurations.
 **  Assumes python module named <detType>_config.py with functions:
 **      <detType>_scan_keys(json_str)
 **      <detType>_update   (json_str)
 **/
#pragma once

#include <Python.h>
#include <nlohmann/json.hpp>
#include "xtcdata/xtc/NamesLookup.hh"

namespace XtcData  {
    class Xtc;
    class NamesId;
}

namespace Drp {
class Parameters;
class PythonConfigScanner
{
public:
    PythonConfigScanner(const Parameters&, PyObject& module);
    ~PythonConfigScanner();
public:
    unsigned configure(const nlohmann::json&    keys,
                       XtcData::Xtc&            xtc,
                       const void*              bufEnd,
                       XtcData::NamesId&        namesId,
                       XtcData::NamesLookup&    namesLookup,
                       std::vector<unsigned>    segNos={},
                       std::vector<std::string> serNos={});
    unsigned step     (const nlohmann::json&    dict,
                       XtcData::Xtc&            xtc,
                       const void*              bufEnd,
                       XtcData::NamesId&        namesId,
                       XtcData::NamesLookup&    namesLookup,
                       std::vector<unsigned>    segNos={},
                       std::vector<std::string> serNos={});
private:
    const Parameters& m_para;
    PyObject&         m_module;
};
};
