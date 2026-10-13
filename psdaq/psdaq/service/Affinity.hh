#ifndef Pds_Affinity_hh
#define Pds_Affinity_hh

#include <string>
#include <vector>
#include <cstdint>

namespace Pds
{
  enum { MAX_NUMA_NODES = 8 };

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
}

#endif
