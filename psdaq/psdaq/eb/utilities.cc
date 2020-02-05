#include "utilities.hh"

#include <stdio.h>
#include <unistd.h>                     // sysconf()
#include <stdlib.h>                     // posix_memalign()
#include <string.h>                     // strerror(), memset()
#include <dlfcn.h>

#include <Python.h>
#include <rapidjson/document.h>

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
    fprintf(stderr, "%s:\n  Error from posix_memalign:\n  %s",
            __PRETTY_FUNCTION__, strerror(ret));
    return nullptr;
  }

//#define VALGRIND                       // Avoid uninitialized memory commentary
#ifdef  VALGRIND                       // Region is initialized by RDMA,
  memset(region, 0, size);             // but Valgrind can't know that
#endif

  return region;
}

int Pds::Eb::pinThread(const pthread_t& th, int cpu)
{
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu, &cpuset);
  return pthread_setaffinity_np(th, sizeof(cpu_set_t), &cpuset);
}
