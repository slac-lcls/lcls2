#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif
#include "Affinity.hh"

#include <sched.h>
#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "psalg/utils/SysLog.hh"
using logging = psalg::SysLog;

using namespace Pds;


namespace Pds {

// PCI address of a datadev, from the first line of /proc/<name>:
//   PCIe[BUS:NUM:SLOT.FUNC] : 0000:04:00.0
// Split after the ']', since the label itself contains colons.  Empty if unavailable,
// which logDeviceLocality() reports rather than treats as an error.
std::string pciAddrOfDatadev(const std::string& device)
{
  auto slash = device.rfind('/');
  std::string path("/proc/" + (slash == std::string::npos ? device
                                                          : device.substr(slash + 1)));
  FILE* f = fopen(path.c_str(), "r");
  if (!f)  return std::string();

  char line[256];
  char pci[64] = {0};
  if (fgets(line, sizeof(line), f)) {
    auto bracket = strchr(line, ']');
    auto sep     = bracket ? strchr(bracket, ':') : nullptr;
    if (sep && sscanf(sep + 1, " %63s", pci) != 1)  pci[0] = '\0';
  }
  fclose(f);
  return std::string(pci);
}

int numaNodeOfDatadev(const std::string& device)
{
  auto pciAddr = pciAddrOfDatadev(device);
  return pciAddr.empty() ? -1 : numaNodeOfPciDev(pciAddr);
}

int numaNodeOfCpu(int cpu)
{
  for (int node = 0; node < MAX_NUMA_NODES; ++node) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpu%d", node, cpu);
    if (!access(path, F_OK))  return node;
  }
  return -1;
}

int numaNodeOfPciDev(const std::string& pciAddr)
{
  std::string path("/sys/bus/pci/devices/" + pciAddr + "/numa_node");
  FILE* f = fopen(path.c_str(), "r");
  if (!f)  return -1;
  int node = -1;
  if (fscanf(f, "%d", &node) != 1)  node = -1;
  fclose(f);
  return node;
}

std::vector<int> cpuAllocation(uint64_t* nodeMask)
{
  std::vector<int> cpus;
  if (nodeMask)  *nodeMask = 0;

  cpu_set_t set;
  if (sched_getaffinity(0, sizeof(set), &set)) {
    logging::error("sched_getaffinity failed: %m");
    return cpus;
  }
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &set))  continue;
    cpus.push_back(cpu);
    int node = numaNodeOfCpu(cpu);
    if (nodeMask && node >= 0 && node < MAX_NUMA_NODES)  *nodeMask |= 1ul << node;
  }
  return cpus;
}

// What `cores:` asked for is a request; this is what the kernel actually granted,
// which is the number to compare a thread count or a device's locality against.
void logCpuAllocation(const char* who)
{
  uint64_t nodeMask;
  auto     cpus = cpuAllocation(&nodeMask);
  if (cpus.empty())  return;

  std::string list, nodes;
  for (auto cpu : cpus)  list  += (list.empty()  ? "" : ",") + std::to_string(cpu);
  for (int n = 0; n < MAX_NUMA_NODES; ++n)
    if (nodeMask & (1ul << n))  nodes += (nodes.empty() ? "" : ",") + std::to_string(n);

  logging::info("%s: granted %zu CPUs on NUMA node%s %s: %s", who, cpus.size(),
                nodes.find(',') != std::string::npos ? "s" : "",
                nodes.c_str(), list.c_str());
}

// Name a device's NUMA node and say whether our CPUs share it.
void logDeviceLocality(const char* who, const char* what, const std::string& pciAddr)
{
  uint64_t nodeMask;
  if (cpuAllocation(&nodeMask).empty())  return;

  if (pciAddr.empty()) {                // Caller could not identify the device
    logging::info("%s: %s has no PCI address, so its locality is unknown", who, what);
    return;
  }
  int node = numaNodeOfPciDev(pciAddr);
  if (node < 0) {
    logging::info("%s: %s at %s has no NUMA node", who, what, pciAddr.c_str());
    return;
  }
  bool local = node < MAX_NUMA_NODES && (nodeMask & (1ul << node));
  logging::info("%s: %s at %s is on NUMA node %d, %s our CPUs",
                who, what, pciAddr.c_str(), node, local ? "local to" : "REMOTE from");
}

unsigned threadCount()
{
  DIR* dir = opendir("/proc/self/task");
  if (!dir)  return 0;
  unsigned count = 0;
  while (auto entry = readdir(dir))
    if (entry->d_name[0] != '.')  ++count;
  closedir(dir);
  return count;
}

// Compare the threads running against the CPUs granted, rather than silently
// timesharing.  Call once the caller's own threads are up.
//
// hotThreads is the caller's count of its spinning threads.  Most threads belong to
// libraries beneath us and most are blocked, so neither number is a recommendation:
// this is a prompt to measure.
void logThreadBudget(const char* who, unsigned hotThreads)
{
  auto cpus    = cpuAllocation();
  auto threads = threadCount();
  if (cpus.empty())  return;

  logging::info("%s: %u threads (%u of them ours and hot) in %zu granted CPUs",
                who, threads, hotThreads, cpus.size());
  if (hotThreads > cpus.size())
    logging::warning("%s: %u hot threads in %zu CPUs: they will timeshare.  "
                     "Consider cores: %u", who, hotThreads, cpus.size(), hotThreads);
}

} // Pds
