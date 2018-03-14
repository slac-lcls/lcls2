#include "EbLfServer.hh"
#include "EbLfClient.hh"

#include "Endpoint.hh"

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

static const unsigned max_peers     = 64;    // Maximum possible number of peers
static const unsigned port_base     = 32768; // Base port number
static const unsigned default_id    = 0;
static const unsigned default_size  = 4096;
static const unsigned rx_depth      =  500;
static const unsigned default_iters = 1000;
static const bool     default_poll  = true;

typedef std::chrono::microseconds us_t;

// Work Completion IDs
enum { PINGPONG_RECV_WCID = 1,
       PINGPONG_SEND_WCID = 2,
};


void usage(char *name, char *desc)
{
  if (desc)
    fprintf(stderr, "\n%s\n", desc);

  fprintf(stderr, "Usage:\n");
  fprintf(stderr, "  %s [OPTIONS]          start server\n", name);
  fprintf(stderr, "  %s [OPTIONS] <host>   connect to server\n", name);

  fprintf(stderr, "\nOptions:\n");

  fprintf(stderr, " %-20s %s (default: %s)\n",    "-A <interface_addr>",
          "IP address of the interface to use",   "libfabric's 'best' choice");
  fprintf(stderr, " %-20s %s (port: %d)\n",       "-P <port>",
          "Base port number",                     port_base);

  fprintf(stderr, " %-20s %s (default: %d)\n",    "-s <size>",
          "Size of message to exchange",          default_size);
  fprintf(stderr, " %-20s %s (default: %d)\n",    "-r <rx-depth>",
          "Number of receives to post at a time", rx_depth);
  fprintf(stderr, " %-20s %s (default: %d)\n",    "-n <iters>",
          "Number of exchanges",                  default_iters);
  fprintf(stderr, " %-20s %s (default: %s)\n",    "-e",
          "Sleep on CQ events",                   default_poll ? "poll" : "sleep");

  fprintf(stderr, " %-20s %s\n", "-h", "display this help output");
}

int main(int argc, char **argv)
{
  int      op, ret  = 0;
  unsigned srcId    = default_id;
  char*    ifAddr   = nullptr;
  unsigned portBase = port_base;
  unsigned size     = default_size;
  unsigned rxDepth  = rx_depth;
  unsigned iters    = default_iters;
  bool     usePoll  = default_poll;

  while ((op = getopt(argc, argv, "h?A:P:s:r:n:e")) != -1)
  {
    switch (op)
    {
      case 'A':  ifAddr   = optarg;        break;
      case 'P':  portBase = atoi(optarg);  break;
      case 's':  size     = atoi(optarg);  break;
      case 'r':  rxDepth  = atoi(optarg);  break;
      case 'n':  iters    = atoi(optarg);  break;
      case 'e':  usePoll  = false;         break;
      case '?':
      case 'h':
      default:
        usage(argv[0], (char*)"EbLfPingpong test");
        return 1;
    }
  }

  if (portBase + srcId > USHRT_MAX)
  {
    fprintf(stderr, "Server port is out of range 0 - 65535: %d\n",
            portBase + srcId);
    return 1;
  }

  char port[8];
  std::string srvPort;
  std::vector<std::string> cltAddr;
  std::vector<std::string> cltPort;
  unsigned dstId = default_id;
  uint64_t peers = 1ul << dstId;
  if (optind == argc - 1)
  {
    char* peer = argv[optind];
    if (portBase + dstId > USHRT_MAX)
    {
      fprintf(stderr, "Peer port is out of range 0 - 65535: %d\n",
              portBase + dstId);
      return 1;
    }
    snprintf(port, sizeof(port), "%d", portBase + dstId);
    cltAddr.push_back(std::string(peer));
    cltPort.push_back(std::string(port));
  }
  else
  {
    snprintf(port, sizeof(port), "%d", portBase + srcId);
    srvPort = std::string(port);
  }

  //if (rxDepth) {
  //  if (rxDepth > fi->rx_attr->size) {
  //    fprintf(stderr, "rxDepth requested: %d, "
  //            "rxDepth supported: %zd\n", rxDepth, fi->rx_attr->size);
  //    rc = 1;
  //    goto err1;
  //  }
  //} else {
  //  rxDepth = (rx_depth > fi->rx_attr->size) ?
  //    fi->rx_attr->size : rx_depth;
  //}

  size_t alignment = sysconf(_SC_PAGESIZE);
  size             = alignment * ((size + alignment - 1) / alignment);
  void*  buffer    = nullptr;
  ret              = posix_memalign(&buffer, alignment, size);
  if (ret)  perror("posix_memalign");
  assert(buffer != nullptr);

  unsigned pending = PINGPONG_RECV_WCID;
  unsigned routs   = 0;

  EbLfServer* server = nullptr;
  EbLfClient* client = nullptr;
  EbLfBase*   transport;
  if (cltAddr.size() == 0)
  {
    server = new EbLfServer(ifAddr, srvPort, std::bitset<64>(peers).count());
    if ( (ret = server->connect(srcId)) )  return ret;
    transport = server;

    if (transport->prepareLclMr(size)) // Sequence must match client's prepareRmtMr()
    {
      fprintf(stderr, "Server: Failed to prepare local memory region\n");
      return 1;
    }

    if (transport->prepareRmtMr(buffer, size)) // Sequence must match client's prepareLclMr()
    {
      fprintf(stderr, "Server: Failed to prepare remote memory region\n");
      return 1;
    }

    routs = transport->postCompRecv(dstId, rxDepth, (void *)(uintptr_t)PINGPONG_RECV_WCID);
    if (routs < rxDepth)
    {
      fprintf(stderr, "Couldn't post receive (%d)\n", routs);
      return 1;
    }
  }
  else
  {
    const unsigned tmo(120);
    client = new EbLfClient(cltAddr, cltPort);
    if ( (ret = client->connect(srcId, tmo)) )  return ret;
    transport = client;

    if (transport->prepareRmtMr(buffer, size)) // Sequence must match server's prepareLclMr()
    {
      fprintf(stderr, "Client: Failed to prepare remote memory region\n");
      return 1;
    }

    if (transport->prepareLclMr(size)) // Sequence must match server's prepareRmtMr()
    {
      fprintf(stderr, "Client: Failed to prepare local memory region\n");
      return 1;
    }

    routs = transport->postCompRecv(dstId, rxDepth, (void *)(uintptr_t)PINGPONG_RECV_WCID);
    if (routs < rxDepth)
    {
      fprintf(stderr, "Couldn't post receive (%d)\n", routs);
      return 1;
    }

    if (transport->post(dstId, buffer, size, 0, 0, (void *)(uintptr_t)PINGPONG_SEND_WCID))
    {
      fprintf(stderr, "Client: failed to post a buffer\n");
      return 1;
    }
    pending |= PINGPONG_SEND_WCID;
  }

  auto tStart = std::chrono::steady_clock::now();
  unsigned rcnt = 0;
  unsigned scnt = 0;
  while ((rcnt < iters || scnt < iters) && (ret == 0))
  {
    fi_cq_data_entry wc;
    if (usePoll)
      ret = transport->pend(&wc);
    else
      ret = transport->pendW(&wc);
    if (ret)
    {
      fprintf(stderr, "Failed pending for a buffer: %s(%d)\n",
              fi_strerror(-ret), ret);
      break;
    }

    switch ((int) (uintptr_t) wc.op_context)
    {
      case PINGPONG_SEND_WCID:
        ++scnt;
        break;

      case PINGPONG_RECV_WCID:
        if (--routs <= 1)
        {
          routs += transport->postCompRecv(dstId, rxDepth - routs, wc.op_context);
          if (routs < rxDepth)
          {
            fprintf(stderr, "Couldn't post receive (%d)\n", routs);
            ret = 1;
            break;
          }
        }

        ++rcnt;
        break;

      default:
        fprintf(stderr, "Completion for unknown wc_id %d\n",
                (int) (uintptr_t) wc.op_context);
        ret = 1;
        break;
    }

    pending &= ~(int) (uintptr_t) wc.op_context;
    if (scnt < iters && !pending)
    {
      if (transport->post(dstId, buffer, size, 0, 0, (void *)(uintptr_t)PINGPONG_SEND_WCID))
      {
        fprintf(stderr, "Failed to post a buffer\n");
        ret = 1;
        break;
      }
      pending = PINGPONG_RECV_WCID | PINGPONG_SEND_WCID;
    }
  }
  auto tEnd = std::chrono::steady_clock::now();

  if (ret == 0)
  {
    double   usecs = std::chrono::duration_cast<us_t>(tEnd - tStart).count();
    uint64_t bytes = (uint64_t)size * iters * 2;
    printf("%ld bytes in %.2f seconds = %.2f Mbit/sec\n",
           bytes, usecs / 1000000., bytes * 8. / usecs);
    printf("%d iters in %.2f seconds = %.2f usec/iter\n",
           iters, usecs / 1000000., usecs / iters);
  }

  if (server)
  {
    server->shutdown();
    delete server;
  }
  if (client)
  {
    client->shutdown();
    delete client;
  }

  return ret;
}
