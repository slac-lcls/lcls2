#include "MonLfServer.hh"
#include "MonLfClient.hh"

#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include <climits>
#include <atomic>


using namespace Pds;
using namespace Pds::Fabrics;
using namespace Pds::Eb;

static const unsigned port_base     = 54321; // Base port number
static const unsigned default_iters = 1000;

static std::atomic<bool> running(true);

void sigHandler( int signal )
{
  static std::atomic<unsigned> callCount(0);

  printf("sigHandler() called: call count = %u\n", callCount.load());

  if (callCount == 0)
  {
    running = false;
  }

  if (callCount++)
  {
    fprintf(stderr, "Aborting on 2nd ^C...\n");
    ::abort();
  }
}

static void usage(char *name, char *desc)
{
  if (desc)
    fprintf(stderr, "%s\n\n", desc);

  fprintf(stderr, "Usage:\n");
  fprintf(stderr, "  %s [OPTIONS] <host>\n", name);

  fprintf(stderr, "\nWhere:\n"
          "  <host> is the IP address of the peer\n");

  fprintf(stderr, "\nOptions:\n");

  fprintf(stderr, " %-20s %s (default: %s)\n",    "-A <interface_addr>",
          "IP address of the interface to use",   "libfabric's 'best' choice");
  fprintf(stderr, " %-20s %s (port: %d)\n",       "-P <port>",
          "Base port number",                     port_base);

  fprintf(stderr, " %-20s %s (default: %d)\n",    "-n <iters>",
          "Number of exchanges",                  default_iters);

  fprintf(stderr, " %-20s %s\n", "-h", "display this help output");
}

int main(int argc, char **argv)
{
  int      op, ret  = 0;
  char*    ifAddr   = nullptr;
  unsigned portBase = port_base;
  unsigned iters    = default_iters;

  while ((op = getopt(argc, argv, "h?A:P:n:S")) != -1)
  {
    switch (op)
    {
      case 'A':  ifAddr   = optarg;        break;
      case 'P':  portBase = atoi(optarg);  break;
      case 'n':  iters    = atoi(optarg);  break;
      case '?':
      case 'h':
      default:
        usage(argv[0], (char*)"Test of MonLf links");
        return 1;
    }
  }

  unsigned srvBase = portBase;
  if (srvBase > USHRT_MAX)
  {
    fprintf(stderr, "Server port is out of range 0 - %u: %d\n",
            USHRT_MAX, srvBase);
    return 1;
  }
  std::string srvPort(std::to_string(srvBase));

  std::vector<std::string> cltAddr;
  std::vector<std::string> cltPort;
  if (optind < argc)
  {
    char*    peer    = argv[optind];
    unsigned cltBase = portBase;
    if (cltBase > USHRT_MAX)
    {
      fprintf(stderr, "Peer port is out of range 0 - %u: %d\n",
              USHRT_MAX, cltBase);
      return 1;
    }
    cltAddr.push_back(std::string(peer));
    cltPort.push_back(std::string(std::to_string(cltBase)));
  }

  MonLfServer* server = nullptr;
  MonLfClient* client = nullptr;
  if (cltAddr.size() == 0)
  {
    server = new MonLfServer(ifAddr, srvPort);
    if ( (ret = server->connect())    )  return ret;
  }
  else
  {
    const unsigned tmo(120);
    client = new MonLfClient(cltAddr, cltPort);
    if ( (ret = client->connect(tmo)) )  return ret;
  }

  if (server)
  {
    ::signal( SIGINT, sigHandler );

    while (running)
    {
      int      rc;
      uint64_t data;
      if ((rc = server->poll(&data)) > 0)
        printf("Got %016lx\n", data);
      else if (rc < 0)
        printf("Error %d\n", rc);

      sleep(1);
    }
  }

  if (client)
  {
    unsigned cnt = 0;
    while (cnt < iters)
    {
      int      rc;
      unsigned dst  = 0;
      uint64_t data = (dst << 24) + cnt;
      printf("Posting %016lx\n", data);
      if ((rc = client->post(dst, data)) == 0)
        ++cnt;
      else
        printf("Error %d\n", rc);
      sleep(1);
    }
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

  return 0;
}
