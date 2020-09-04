/**
 **  Class to support scanning of python-implemented detector configurations.
 **  Assumes python module named <detType>_config.py with functions:
 **      <detType>_scan_keys(json_str)
 **      <detType>_update   (json_str)
 **/
#pragma once

#include <Python.h>
#include "xtcdata/xtc/NamesLookup.hh"
#include "psdaq/service/json.hpp"

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
    unsigned configure(const nlohmann::json& keys, 
                       XtcData::Xtc&         xtc,
                       XtcData::NamesId&     namesId,
                       XtcData::NamesLookup& namesLookup);
    unsigned step     (const nlohmann::json& dict, 
                       XtcData::Xtc&         xtc,
                       XtcData::NamesId&     namesId,
                       XtcData::NamesLookup& namesLookup);
private:
    const Parameters& m_para;
    PyObject&         m_module;
};
};
