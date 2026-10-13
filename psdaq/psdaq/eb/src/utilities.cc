#include "utilities.hh"
#include "Endpoint.hh"

#include <stdio.h>
#include <unistd.h>                     // sysconf()
#include <stdlib.h>                     // posix_memalign()
#include <string.h>                     // strerror(), memset()
#include <dlfcn.h>
#include <limits.h>                    // PATH_MAX

#include <Python.h>
#include <rapidjson/document.h>

#include "psdaq/service/Affinity.hh"   // logDeviceLocality
#include "psalg/utils/SysLog.hh"
using logging = psalg::SysLog;

using namespace Pds::Eb;
using namespace rapidjson;


// PCI address of a libfabric domain, which names either an RDMA device (mlx5_0) or a
// netdev (enp73s0f0) depending on the provider.  Empty if neither resolves.
static std::string _devicePciAddr(const char* domain)
{
  if (!domain || !*domain)  return std::string();

  for (auto dir : {"/sys/class/infiniband/", "/sys/class/net/"}) {
    std::string link(std::string(dir) + domain + "/device");
    char        path[PATH_MAX];
    if (auto p = realpath(link.c_str(), path)) {
      std::string resolved(p);
      return resolved.substr(resolved.rfind('/') + 1);
    }
  }
  return std::string();
}

void Pds::Eb::logFabricInfo(const char* who, Pds::Fabrics::Fabric* fabric)
{
  static bool logged = false;           // Once per process, not once per link
  if (logged || !fabric)  return;

  auto info = fabric->info();
  if (!info)  return;                   // Before latching, or nothing is ever logged

  logged = true;

  // Unconditional, not under -v: a run whose log lacks this cannot be diagnosed
  // after the fact, and it is a few lines once.
  logging::info("%s: libfabric %s, provider '%s' %u.%u, domain '%s', fabric '%s'",
                who, fi_tostr(info, FI_TYPE_VERSION), fabric->provider(),
                FI_MAJOR(info->fabric_attr->prov_version),
                FI_MINOR(info->fabric_attr->prov_version),
                fabric->domain_name(), fabric->fabric_name());

  // Negotiated, not requested: the hints ask for 0, so these are the provider's
  // own values and rx_attr->size bounds the Rx CQ.
  logging::info("%s:   rx_size %zu, tx_size %zu, inject_size %zu, cq_data_size %zu",
                who, info->rx_attr->size, info->tx_attr->size,
                info->tx_attr->inject_size, info->domain_attr->cq_data_size);

  // One fi_tostr() per logging call: it returns a pointer to a single static
  // buffer, so two in the same statement both render the last one evaluated.
  //
  // We never request a progress mode, so this is whichever fi_getinfo returned
  // first, and that ordering is not contractual.  MANUAL would couple all
  // provider progress to the poller's call rate.
  logging::info("%s:   data_progress %s", who,
                fi_tostr(&info->domain_attr->data_progress,    FI_TYPE_PROGRESS));
  logging::info("%s:   control_progress %s", who,
                fi_tostr(&info->domain_attr->control_progress, FI_TYPE_PROGRESS));
  logging::info("%s:   mr_mode %s", who,
                fi_tostr(&info->domain_attr->mr_mode,          FI_TYPE_MR_MODE));

  // FI_ORDER_RMA_WAW is what makes posting order the order writes land in the
  // Results region, which the orphaned-Input trap's reasoning depends on.
  logging::info("%s:   msg_order %s", who,
                fi_tostr(&info->tx_attr->msg_order, FI_TYPE_MSG_ORDER));

  // A network filesystem may share this device, so its locality is not only ours.
  auto pci = _devicePciAddr(fabric->domain_name());
  if (!pci.empty())  logDeviceLocality(who, "fabric device", pci);
}


size_t Pds::Eb::roundUpSize(size_t size)
{
  size_t pageSize = sysconf(_SC_PAGESIZE);
  return pageSize * ((size + pageSize - 1) / pageSize);
}

void* Pds::Eb::allocRegion(size_t size)
{
  size_t pageSize = sysconf(_SC_PAGESIZE);
  void*  region   = nullptr;
  int    ret      = posix_memalign(&region, pageSize, roundUpSize(size));
  if (ret)
  {
    fprintf(stderr, "%s:\n  Error from posix_memalign:\n  %s\n",
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
  if (cpu > 0)
  {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    return pthread_setaffinity_np(th, sizeof(cpu_set_t), &cpuset);
  }
  return 0;
}
