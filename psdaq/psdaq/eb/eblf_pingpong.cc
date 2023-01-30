#include "EbLfServer.hh"
#include "EbLfClient.hh"

#include "EbLfLink.hh"
#include "utilities.hh"

#include "psdaq/service/kwargs.hh"
#include "psdaq/epicstools/PvMonitorBase.hh"

#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif
#include <sched.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include <climits>
#include <bitset>
#include <chrono>
#include <atomic>
#include <thread>


using namespace Pds;
using namespace Pds::Fabrics;
using namespace Pds::Eb;

using kwmap_t  = std::map<std::string,std::string>;

static const unsigned port_base     = 1024; // Pick from range 1024 - 32768, 61000 - 65535
static const unsigned default_size  = 4096;
static const unsigned default_iters = 1000;
static const int      default_core  = -1;

typedef std::chrono::microseconds us_t;


void usage(char *name, char *desc)
{
  if (desc)
    fprintf(stderr, "%s\n\n", desc);

  fprintf(stderr, "Usage:\n");
  fprintf(stderr, "  %s [OPTIONS] <host>\n", name);

  fprintf(stderr, "\nWhere:\n"
          "  <host> is the IP address of the replying side\n");

  fprintf(stderr, "\nOptions:\n");

  fprintf(stderr, " %-20s %s (default: %s)\n",    "-A <interface_addr>",
          "IP address of the interface to use",   "libfabric's 'best' choice");
  fprintf(stderr, " %-20s %s (port: %d)\n",       "-P <port>",
          "Base port number",                     port_base);

  fprintf(stderr, " %-20s %s (default: %d)\n",    "-s <size>",
          "Size of message to exchange",          default_size);
  fprintf(stderr, " %-20s %s (default: %d)\n",    "-n <iters>",
          "Number of exchanges",                  default_iters);
  fprintf(stderr, " %-20s %s (default: %d)\n",    "-c <core>",
          "CPU core to run on",                   default_core);
  fprintf(stderr, " %-20s %s\n",                  "-E",
          "Enables testing with EPICS by monitoring a PV");
  fprintf(stderr, " %-20s %s\n",                  "-S",
          "Start flag: one side must specify this");
  fprintf(stderr, " %-20s %s\n",                  "-v",
          "Verbose flag");

  fprintf(stderr, " %-20s %s\n", "-h", "display this help output");
}

int main(int argc, char **argv)
{
  int         op, rc   = 0;
  std::string ifAddr   = { };
  unsigned    portBase = port_base;
  size_t      size     = default_size;
  unsigned    iters    = default_iters;
  int         core     = default_core;
  bool        epics    = false;
  bool        start    = false;
  unsigned    verbose  = 0;
  std::string kwargs_str;
  kwmap_t     kwargs;

  while ((op = getopt(argc, argv, "h?A:P:s:r:n:c:k:ESv")) != -1)
  {
    switch (op)
    {
      case 'A':  ifAddr     = optarg;                      break;
      case 'P':  portBase   = atoi(optarg);                break;
      case 's':  size       = atoi(optarg);                break;
      case 'n':  iters      = atoi(optarg);                break;
      case 'c':  core       = atoi(optarg);                break;
      case 'E':  epics      = true;                        break;
      case 'S':  start      = true;                        break;
      case 'k':  kwargs_str = kwargs_str.empty()
                            ? optarg
                            : kwargs_str + ", " + optarg;  break;
      case 'v':  verbose    = 1;                           break;
      case '?':
      case 'h':
      default:
        usage(argv[0], (char*)"EbLfPingpong sends 'pings' from one port and "
                              "receives 'pongs' on an adjacent port.");
        return 1;
    }
  }

  unsigned srvBase = portBase + (start ? 1 : 0);
  if (srvBase > USHRT_MAX)
  {
    fprintf(stderr, "Server port is out of range 0 - %u: %d\n",
            USHRT_MAX, srvBase);
    return 1;
  }
  std::string srvPort(std::to_string(srvBase));

  std::vector<std::string> cltAddr(1);
  std::vector<std::string> cltPort(1);
  if (optind < argc)
  {
    char*    peer    = argv[optind];
    unsigned cltBase = portBase + (start ? 0 : 1);
    if (cltBase > USHRT_MAX)
    {
      fprintf(stderr, "Peer port is out of range 0 - %u: %d\n",
              USHRT_MAX, cltBase);
      return 1;
    }
    cltAddr[0] = std::string(peer);
    cltPort[0] = std::string(std::to_string(cltBase));
  }
  else
  {
    fprintf(stderr, "Peer address is required\n");
    return 1;
  }

  get_kwargs(kwargs_str, kwargs);
  for (const auto& kwargs : kwargs)
  {
    if (kwargs.first == "ep_fabric")    continue;
    if (kwargs.first == "ep_domain")    continue;
    if (kwargs.first == "ep_provider")  continue;
    fprintf(stderr, "Unrecognized kwarg '%s=%s'\n",
            kwargs.first.c_str(), kwargs.second.c_str());
    return 1;
  }

  if (core != -1)
  {
    rc = pinThread(pthread_self(), core);
    if (rc != 0)
    {
      fprintf(stderr, "%s:\n  Error from pinThread:\n  %s\n",
              __PRETTY_FUNCTION__, strerror(rc));
    }
  }

  size_t alignment = sysconf(_SC_PAGESIZE);
  size_t srcSize   = alignment * ((size + alignment - 1) / alignment);
  size_t snkSize;
  void*  snkBuf    = nullptr;
  void*  srcBuf    = nullptr;
  rc               = posix_memalign(&srcBuf, alignment, srcSize);
  if (rc)
  {
    fprintf(stderr, "posix_memalign failed for source buffer: %s", strerror(rc));
    return rc;
  }

  // For testing to see if there is a conflict with EPICS:
  class PvMon : public Pds_Epics::PvMonitorBase
  {
  public:
    PvMon(const std::string& channelName,
          const std::string& provider = "pva") :
      Pds_Epics::PvMonitorBase(channelName, provider)
    {
    }
  public:
    void onConnect()    override {};
    void onDisconnect() override {};
    void updated()      override {};
  };

  PvMon* pvMon = nullptr;

  unsigned     id      = 0;
  EbLfServer*  srv     = nullptr;
  EbLfClient*  clt     = nullptr;
  unsigned     nLinks  = 1;
  std::vector<EbLfSvrLink*> srvLinks(nLinks);
  std::vector<EbLfCltLink*> cltLinks(nLinks);
  if (!start)
  {
    srv = new EbLfServer(verbose, kwargs);
    if ( (rc = linksStart(*srv, ifAddr, srvPort, nLinks, "Srv")) )
    {
      fprintf(stderr, "Failed to initialize EbLfServer\n");
      return rc;
    }
    if ( (rc = linksConnect(*srv, srvLinks, id, "Srv")) )
    {
      fprintf(stderr, "Error connecting to client\n");
      return rc;
    }
    if ( (rc = srvLinks[0]->prepare( &snkSize, "Srv")) < 0 )
    {
      fprintf(stderr, "Failed to prepare server's link\n");
      return rc;
    }
    rc = posix_memalign(&snkBuf, alignment, snkSize);
    if (rc)
    {
      fprintf(stderr, "posix_memalign failed for sink buffer: %s", strerror(rc));
      return rc;
    }
    if ( (rc = srvLinks[0]->setupMr(snkBuf, snkSize, "Srv")) )
    {
      fprintf(stderr, "Failed to set up MemoryRegion\n");
      return rc;
    }
    printf("EbLfClient ID %d connected to server\n", srvLinks[0]->id());

    clt = new EbLfClient(verbose, kwargs);
    if ( (rc = linksConnect(*clt, cltLinks, cltAddr, cltPort, id, "Srv")) )
    {
      fprintf(stderr, "Error connecting to server\n");
      return rc;
    }
    if ( (rc = linksConfigure(cltLinks, srcBuf, srcSize, "Srv")) < 0)
    {
      fprintf(stderr, "Failed to prepare client's link\n");
      return rc;
    }
    printf("EbLfServer ID %d connected to client\n", cltLinks[0]->id());
  }
  else
  {
    if (epics)
      pvMon = new PvMon("EM1K0:GMD:HPS:STR0:STREAM_SHORT2", "ca");

    clt = new EbLfClient(verbose, kwargs);
    if ( (rc = linksConnect(*clt, cltLinks, cltAddr, cltPort, id, "Clt")) )
    {
      fprintf(stderr, "Error connecting to server\n");
      return rc;
    }
    if ( (rc = linksConfigure(cltLinks, srcBuf, srcSize, "Clt")) < 0)
    {
      fprintf(stderr, "Failed to prepare client's link\n");
      return rc;
    }
    printf("EbLfServer ID %d connected to client\n", cltLinks[0]->id());

    srv = new EbLfServer(verbose, kwargs);
    if ( (rc = linksStart(*srv, ifAddr, srvPort, nLinks, "Clt")) )
    {
      fprintf(stderr, "Failed to initialize EbLfServer\n");
      return rc;
    }
    if ( (rc = linksConnect(*srv, srvLinks, id, "Clt")) )
    {
      fprintf(stderr, "Error connecting to client\n");
      return rc;
    }
    if ( (rc = srvLinks[0]->prepare(&snkSize, "Clt")) < 0)
    {
      fprintf(stderr, "Failed to prepare server's link\n");
      return rc;
    }
    rc = posix_memalign(&snkBuf, alignment, snkSize);
    if (rc)
    {
      fprintf(stderr, "posix_memalign failed for sink buffer: %s", strerror(rc));
      return rc;
    }
    if ( (rc = srvLinks[0]->setupMr(snkBuf, snkSize, "Clt")) )
    {
      fprintf(stderr, "Failed to set up MemoryRegion\n");
      return rc;
    }
    printf("EbLfClient ID %d connected to server\n", srvLinks[0]->id());

    if (cltLinks[0]->post(srcBuf, srcSize, 0, 0))
    {
      fprintf(stderr, "Clt: failed to post a buffer\n");
      return 1;
    }
  }

  auto tStart = std::chrono::steady_clock::now();
  unsigned rcnt = 0;
  unsigned scnt = start ? 1 : 0;
  while (rcnt < iters || scnt < iters)
  {
    fi_cq_data_entry wc;
    const int msTmo = 5000;
    if ( (rc = srv->pend(&wc, msTmo)) < 0 )
    {
      fprintf(stderr, "Failed pending for a buffer: %s(%d)\n",
              fi_strerror(-rc), rc);
      break;
    }
    rc = 0;
    ++rcnt;

    if (scnt < iters)
    {
      if ( (rc = cltLinks[0]->post(srcBuf, srcSize, 0, 0)) )
      {
        fprintf(stderr, "Failed to post a buffer: %s(%d)\n",
                fi_strerror(-rc), rc);
        break;
      }
      ++scnt;
    }
  }
  auto tEnd = std::chrono::steady_clock::now();

  if (rc == 0)
  {
    double usecs = std::chrono::duration_cast<us_t>(tEnd - tStart).count();
    size_t bytes = (srcSize + snkSize) * iters;
    printf("%zd bytes in %.2f seconds = %.2f Mbit/sec\n",
           bytes, usecs / 1000000., bytes * 8. / usecs);
    printf("%d iters in %.2f seconds = %.2f usec/iter\n",
           iters, usecs / 1000000., usecs / iters);
  }
  else
  {
    fprintf(stderr, "Exiting with error %d\n", rc);
  }

  if (srv)
  {
    srv->disconnect(srvLinks[0]);
    delete srv;
  }
  if (clt)
  {
    clt->disconnect(cltLinks[0]);
    delete clt;
  }
  free(snkBuf);
  free(srcBuf);

  if (pvMon)
  {
    printf("pvMon: name %s, connected %d\n", pvMon->name().c_str(), pvMon->connected());
    delete pvMon;
  }

  return rc;
}
