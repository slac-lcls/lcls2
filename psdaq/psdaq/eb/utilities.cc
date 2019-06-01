#include "utilities.hh"

#include <stdio.h>
#include <unistd.h>                     // sysconf()
#include <stdlib.h>                     // posix_memalign()
#include <string.h>                     // strerror(), memset()
#include <dlfcn.h>

#include <Python.h>
#include "rapidjson/document.h"

using namespace Pds::Eb;
using namespace rapidjson;


size_t Pds::Eb::roundUpSize(size_t size)
{
  size_t pageSize = sysconf(_SC_PAGESIZE);
  return pageSize * ((size + pageSize - 1) / pageSize);
}

void* Pds::Eb::allocRegion(size_t size)
{
  size_t pageSize = sysconf(_SC_PAGESIZE);
  void*  region   = nullptr;
  int    ret      = posix_memalign(&region, pageSize, size);
  if (ret)
  {
    fprintf(stderr, "posix_memalign failed: %s", strerror(ret));
    return nullptr;
  }

//#define VALGRIND                       // Avoid uninitialized memory commentary
#ifdef  VALGRIND                       // Region is initialized by RDMA,
  memset(region, 0, size);             // but Valgrind can't know that
#endif

  return region;
}

void Pds::Eb::pinThread(const pthread_t& th, int cpu)
{
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu, &cpuset);
  int rc = pthread_setaffinity_np(th, sizeof(cpu_set_t), &cpuset);
  if (rc != 0)
  {
    fprintf(stderr, "Error calling pthread_setaffinity_np: %d\n  %s\n",
            rc, strerror(rc));
  }
}

static int check(PyObject* obj)
{
  if (!obj)
  {
    PyErr_Print();
    return -1;
  }
  return 0;
}

// Note: This function requires a higher layer to call
// Py_Initialize() / Py_Finalize() as appropriate.
int Pds::Eb::fetchFromCfgDb(const std::string& detName,
                            Document&          top,
                            const std::string& connect_json)
{
  int rc = 0;

  // returns new reference
  PyObject* pModule = PyImport_ImportModule("psalg.configdb.get_config");
  if (check(pModule))  return -1;
  // returns borrowed reference
  PyObject* pDict = PyModule_GetDict(pModule);
  if (check(pDict))  return -1;
  // returns borrowed reference
  PyObject* pFunc = PyDict_GetItemString(pDict, (char*)"get_config_json");
  if (check(pFunc))  return -1;
  // returns new reference
  // FIXME: need to get "BEAM" string from config phase1
  PyObject* mybytes = PyObject_CallFunction(pFunc,"sss",connect_json.c_str(), "BEAM", detName.c_str());
  if (check(mybytes))  return -1;
  // returns new reference
  PyObject * json_bytes = PyUnicode_AsASCIIString(mybytes);
  if (check(json_bytes))  return -1;
  char* json = (char*)PyBytes_AsString(json_bytes);

  // to be concise, this doesn't do the rapidjson type-checking
  if (top.Parse(json).HasParseError())
  {
    fprintf(stderr, "%s: json parse error\n", __PRETTY_FUNCTION__);
    rc = -1;
  }

  Py_DECREF(pModule);
  Py_DECREF(mybytes);
  Py_DECREF(json_bytes);

  return rc;
}
