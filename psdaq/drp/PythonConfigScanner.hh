/**
 * @file
 * @brief PythonConfigScanner, which turns scan keys and step values from a detector Python module into XTC configuration and update data.
 */
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
/** Scan support for Python-configured detectors: calls the functions `{detType}_scan_keys` and `{detType}_update` of the given module (per the header comment) and translates their JSON into XTC. */
class PythonConfigScanner
{
public:
    /** Keep references to the parameters and the Python module. */
    PythonConfigScanner(const Parameters&, PyObject& module);
    /** Does nothing; empty body. */
    ~PythonConfigScanner();
public:
    /** Call `{detType}_scan_keys` with keys, translate the returned JSON (or list of JSON, one per segment) into XTC Names with namesId + segment (segment numbers from detName:RO or from segNos, with serNos), and append them to xtc. Returns 0, or -1 (as unsigned) if translation fails; throws a C string on a Python error or an oversize result. */
    unsigned configure(const nlohmann::json&    keys,
                       XtcData::Xtc&            xtc,
                       const void*              bufEnd,
                       XtcData::NamesId&        namesId,
                       XtcData::NamesLookup&    namesLookup,
                       std::vector<unsigned>    segNos={},
                       std::vector<std::string> serNos={});
    /** Call `{detType}_update` with dict, translate the returned JSON (or list of JSON, one per segment) into XTC, and append it to xtc. Returns 0, or -1 (as unsigned) if translation fails; throws a C string on a Python error or an oversize result. */
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
