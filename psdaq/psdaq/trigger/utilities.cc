#include "utilities.hh"

#include <Python.h>
#include <rapidjson/document.h>

using namespace rapidjson;


static bool failed(PyObject* obj)
{
  if (!obj)
  {
    PyErr_Print();
    return true;
  }
  return false;
}

// Note: This function requires a higher layer to call
// Py_Initialize() / Py_Finalize() as appropriate.
int Pds::Trg::fetchDocument(const std::string& connectMsg,
                            const std::string& configAlias,
                            const std::string& detName,
                            Document&          top)
{
  int rc = -1;

  // returns new reference
  PyObject* pModule = PyImport_ImportModule("psalg.configdb.get_config");
  if (!failed(pModule))
  {
    // returns borrowed reference
    PyObject* pDict = PyModule_GetDict(pModule);
    if (!failed(pDict))
    {
      // returns borrowed reference
      PyObject* pFunc = PyDict_GetItemString(pDict, (char*)"get_config_json");
      if (!failed(pFunc))
      {
        // returns new reference
        PyObject* mybytes = PyObject_CallFunction(pFunc, "sssi", connectMsg.c_str(), configAlias.c_str(), detName.c_str(), 0);
        if (!failed(mybytes))
        {
          // returns new reference
          PyObject * json_bytes = PyUnicode_AsASCIIString(mybytes);
          if (!failed(json_bytes))
          {
            char* json = (char*)PyBytes_AsString(json_bytes);

            // to be concise, this doesn't do the rapidjson type-checking
            if (!top.Parse(json).HasParseError())
            {
              rc = 0;
            }
            else
            {
              fprintf(stderr, "%s: json parse error\n", __PRETTY_FUNCTION__);
            }
            Py_DECREF(json_bytes);
          }
          Py_DECREF(mybytes);
        }
      }
    }
    Py_DECREF(pModule);
  }

  return rc;
}
