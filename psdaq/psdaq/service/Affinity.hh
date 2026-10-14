#ifndef Pds_Affinity_hh
#define Pds_Affinity_hh

#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif
#include <pthread.h>
#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace Pds
{
  enum { MAX_NUMA_NODES = 8 };

  // Each DRP's spinning threads, for logThreadBudget():
  //   CPU DRP: PGPreader, Collector, TEBreceiver, plus one per worker
  //   GPU DRP: TigRcvr,   Collector, TEBreceiver, Recorder
  enum { DRP_HOT_THREADS     = 3 };     // Add nworkers for the CPU DRP
  enum { GPU_DRP_HOT_THREADS = 4 };

  // Pin a thread to one absolute CPU; a no-op for cpu <= 0, which is how the `core`
  // options express "unpinned".  Prefer pinHotThread() over naming a CPU directly.
  int pinThread(const pthread_t& th, int cpu);

  // The CPUs this process was granted, and a mask of the NUMA nodes they lie on.
  // This is what the kernel gave us, as against what `cores:` requested.
  std::vector<int> cpuAllocation(uint64_t* nodeMask = nullptr);
  int              numaNodeOfCpu(int cpu);
  int              numaNodeOfPciDev(const std::string& pciAddr);

  // A datadev, named by its device path, e.g. /dev/datadev_04
  std::string      pciAddrOfDatadev(const std::string& device);
  int              numaNodeOfDatadev(const std::string& device);

  unsigned         threadCount();

  void logCpuAllocation(const char* who);

  // Name a device's NUMA node and say whether this process's CPUs share it.
  void logDeviceLocality(const char* who, const char* what, const std::string& pciAddr);

  // Say whether the hot threads fit the allocation.  hotThreads is a floor: most
  // threads are created by libraries beneath us, and most are blocked, not hot.
  void logThreadBudget(const char* who, unsigned hotThreads);

  // Decide this process's pinning, from the `pinThreads` kwarg and what will fit.
  // Call at startup, before any thread that pins itself is created.  nFixedHot is the
  // caller's count of always-hot threads, nWorkers the size of its worker pool, 0 if
  // it has none.
  void setThreadPinning(const std::map<std::string,std::string>& kwargs,
                        unsigned nFixedHot, unsigned nWorkers);

  // Pin the caller to a granted CPU no other caller has taken.  Which thread lands
  // where is arbitrary; only spreading matters.
  void pinHotThread(const char* who);

  // As pinHotThread(), but for a worker out of the pool, which setThreadPinning()
  // leaves unpinned when the pool cannot be given a CPU each.
  void pinWorkerThread(const char* who);
}

#endif
