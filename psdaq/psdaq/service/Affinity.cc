#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif
#include "Affinity.hh"

#include <sched.h>
#include <dirent.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <atomic>

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

// The lowest CPU sharing this one's physical core, which names the core: with SMT off
// that is the CPU itself.  -1 if sysfs cannot say.
int coreOfCpu(int cpu)
{
  char path[128];
  snprintf(path, sizeof(path),
           "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
  FILE* f = fopen(path, "r");
  if (!f)  return -1;

  int  core = -1;
  char list[256] = {0};
  if (fgets(list, sizeof(list), f)) {
    // "6,70" or "6-7"; the first number is the lowest, both forms being sorted
    if (sscanf(list, "%d", &core) != 1)  core = -1;
  }
  fclose(f);
  return core;
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

// Which CPUs may be used is an external policy -- Slurm's CpuSpecList reserves 0-3 on
// the DRP nodes -- so CPU 0 is honoured here rather than reserved for the OS.
int pinThread(const pthread_t& th, int cpu)
{
  if (cpu >= 0)
  {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    return pthread_setaffinity_np(th, sizeof(cpu_set_t), &cpuset);
  }
  return 0;
}

// The granted CPUs, reordered so that one per physical core comes first and the
// hyperthread siblings follow.  Indexing this spreads over distinct cores before it
// doubles up, which plain ascending order does not when SMT is on: a set like
// 6,7,70,71 is only two cores, 6 and 7, since 70 and 71 are their siblings.
static std::vector<int> _spreadOrder(const std::vector<int>& cpus)
{
  std::vector<int> first, siblings, coresTaken;

  for (auto cpu : cpus) {
    int  core  = coreOfCpu(cpu);
    bool taken = false;
    for (auto c : coresTaken)  if (c == core)  taken = true;
    if (core < 0 || !taken) {
      first.push_back(cpu);
      if (core >= 0)  coresTaken.push_back(core);
    }
    else  siblings.push_back(cpu);
  }
  if (!siblings.empty())
    logging::warning("SMT is on: the %zu granted CPUs are only %zu physical cores, "
                     "so any hot thread past the first %zu shares a core",
                     cpus.size(), first.size(), first.size());

  first.insert(first.end(), siblings.begin(), siblings.end());
  return first;
}

static bool             _pinThreads = true;  // On unless the kwarg turns it off
static bool             _pinWorkers = true;  // Cleared when the pool cannot be spread
static std::vector<int> _pinOrder;            // Granted CPUs, best-spread first

// On by default so that a new node or detector gets the placement without an edit to
// every .cnf.py, and `pinThreads=no` is the one knob to turn it off while debugging.
//
// Workers are dropped rather than crammed in: the fixed few gain most from a stable
// CPU, while a pool that outnumbers the allocation would get a fixed round-robin
// where the scheduler can at least migrate.  Pinning cannot help what does not fit.
void setThreadPinning(const std::map<std::string,std::string>& kwargs,
                      unsigned nFixedHot, unsigned nWorkers)
{
  auto kwarg = kwargs.find("pinThreads");
  if (kwarg != kwargs.end()) {
    if      (kwarg->second == "no")      _pinThreads = false;
    else if (kwarg->second == "spread")  _pinThreads = true;
    else {
      logging::critical("Unrecognized pinThreads value '%s': expected 'spread' or 'no'",
                        kwarg->second.c_str());
      abort();
    }
  }
  if (!_pinThreads) {
    logging::info("Thread pinning is disabled");
    return;
  }

  _pinOrder = _spreadOrder(cpuAllocation());

  auto granted = _pinOrder.size();
  if (nWorkers && nFixedHot + nWorkers > granted) {
    _pinWorkers = false;
    logging::warning("%u hot threads (%u of them workers) exceed %zu granted CPUs, "
                     "so the workers are left unpinned.  Consider cores: %u",
                     nFixedHot + nWorkers, nWorkers, granted, nFixedHot + nWorkers);
  }
  if (nFixedHot > granted)
    logging::warning("%u threads to pin in %zu granted CPUs, so some will share.  "
                     "Consider cores: %u", nFixedHot, granted, nFixedHot);
}

// An index into the granted set, never an absolute CPU: Slurm chooses the set and may
// move the job to another socket.  The index comes from a counter so that callers in
// different libraries need no shared numbering.
void pinHotThread(const char* who)
{
  if (!_pinThreads)  return;

  if (_pinOrder.empty())  return;       // No allocation, or setThreadPinning not called

  static std::atomic<unsigned> next{0};
  unsigned                     k = next++;

  int cpu = _pinOrder[k % _pinOrder.size()];
  int rc  = pinThread(pthread_self(), cpu);
  if (rc) {
    logging::error("%s: failed to pin to CPU %d: %s", who, cpu, strerror(rc));
    return;
  }
  logging::info("%s: pinned to CPU %d", who, cpu);
}

void pinWorkerThread(const char* who)
{
  if (_pinWorkers)  pinHotThread(who);
}

} // Pds
