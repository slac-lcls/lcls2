#include "psdaq/eb/utilities.hh"

#include <stdio.h>
#include <unistd.h>                     // sysconf()
#include <stdlib.h>                     // posix_memalign()
#include <string.h>                     // strerror()

using namespace Pds::Eb;


size_t Pds::Eb::roundUpSize(size_t size)
{
  size_t alignment = sysconf(_SC_PAGESIZE);
  return alignment * ((size + alignment - 1) / alignment);
}

void* Pds::Eb::allocRegion(size_t size)
{
  size_t alignment = sysconf(_SC_PAGESIZE);
  void*  region    = nullptr;
  int    ret       = posix_memalign(&region, alignment, size);
  if (ret)
  {
    fprintf(stderr, "posix_memalign failed: %s", strerror(ret));
    return nullptr;
  }

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
