#pragma once

#include <Python.h>
#include <nlohmann/json.hpp>

#include <cstring>
#include <string>
#include <vector>

#include "psalg/utils/SysLog.hh"

namespace Drp {
  namespace Gpu {

// The psana gain modes, (FHG FMG FLG1 FLG2 AHLG1 AHLG2 AMLG1 AMLG2), that gain bit
// values 0 and 1 select.  @todo: Placeholder until the 3x2's mapping is known.
static constexpr unsigned EpixUHRgainModes[]{0, 1};

// Fills cons, laid out [nModes][336*576], with the panel's constants for the run, from
// psdaq.configdb.epixuhr_calib's epixuhr_<what>, e.g. "pedestals" or "gain_factors".
// modes indexes psana's gain modes.  Appends messages for the operator to warns.  The
// caller must hold the GIL.  Returns non-zero on failure.
inline int fetchEpixUHRconstants(const char* what, const std::string& serNo,
                                 const nlohmann::json& runInfo,
                                 const unsigned* modes, unsigned nModes,
                                 std::vector<float>& cons,
                                 std::vector<std::string>& warns)
{
  using logging = psalg::SysLog;

  std::string exp;
  unsigned    run{0};
  if (runInfo.contains("run_info")) {
    auto const& info = runInfo["run_info"];
    if (info.contains("experiment_name"))  exp = info["experiment_name"];
    if (info.contains("run_number"))       run = info["run_number"];
  }

  auto const failed = std::string("no ") + what + " loaded";
  auto module = PyImport_ImportModule("psdaq.configdb.epixuhr_calib");
  if (!module) {
    PyErr_Print();
    warns.push_back(failed + ": psdaq.configdb.epixuhr_calib failed to import");
    return 1;
  }
  auto pyModes = PyList_New(nModes);
  for (unsigned i = 0; i < nModes; ++i)
    PyList_SET_ITEM(pyModes, i, PyLong_FromUnsignedLong(modes[i]));
  auto const func = std::string("epixuhr_") + what;
  auto result = PyObject_CallMethod(module, func.c_str(), "ssIO",
                                    serNo.c_str(), exp.c_str(), run, pyModes);
  Py_DECREF(pyModes);
  Py_DECREF(module);
  if (!result || !PyTuple_Check(result) || PyTuple_Size(result) != 2) {
    if (!result)  PyErr_Print();
    warns.push_back(failed + ": " + func + " raised or returned an unexpected type");
    Py_XDECREF(result);
    return 1;
  }
  auto msgs = PyTuple_GetItem(result, 1);         // Borrowed
  if (PyList_Check(msgs)) {
    for (Py_ssize_t i = 0; i < PyList_Size(msgs); ++i) {
      auto msg = PyUnicode_AsUTF8(PyList_GetItem(msgs, i));
      if (msg)  warns.push_back(msg);
    }
  }
  int err{1};
  auto bytes = PyTuple_GetItem(result, 0);        // Borrowed
  if (PyBytes_Check(bytes) && size_t(PyBytes_Size(bytes)) == cons.size() * sizeof(cons[0])) {
    memcpy(cons.data(), PyBytes_AsString(bytes), cons.size() * sizeof(cons[0]));
    logging::info("Gpu::EpixUHR: loaded %s of %s for %s run %u",
                  what, serNo.c_str(), exp.c_str(), run);
    err = 0;
  } else {
    logging::error("Gpu::EpixUHR: no %s loaded for %s", what, serNo.c_str());
    if (bytes != Py_None)  warns.push_back(failed + ": " + func + " returned the wrong size");
  }
  Py_DECREF(result);
  return err;
}

  } // Gpu
} // Drp
