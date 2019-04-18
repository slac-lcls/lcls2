#include "utilities.hh"

#include <stdio.h>
#include <unistd.h>                     // sysconf()
#include <stdlib.h>                     // posix_memalign()
#include <string.h>                     // strerror(), memset()

using namespace Pds::Eb;


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
