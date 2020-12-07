// This program was a test for the EbLfServer, Client and Link classes.
// It was also used to explore the idea of having two RDMA spaces, one for
// batch data and one for transitions.

// The program is working properly when the output on the Server and the Client
// match.  The sequence goes through the transitions from Unknown to Enabled,
// then <numEvents> L1As, then the transitions from Diabled to to Reset.  This
// repeates <iters> times.

#include "EbLfServer.hh"
#include "EbLfClient.hh"

#include "utilities.hh"

#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/TransitionId.hh"

#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <climits>
#include <atomic>
#include <string>


using namespace XtcData;
using namespace Pds;
using namespace Pds::Fabrics;
using namespace Pds::Eb;

static const unsigned max_servers      = 64;
static const unsigned max_clients      = 64;
static const unsigned port_base        = 54321; // Base port number
static const unsigned default_iters    = 1000;
static const unsigned default_id       = 1;
static const unsigned default_num_evts = 5;
static const unsigned default_buf_cnt  = 10;
static const unsigned default_buf_size = 4096;

static std::atomic<bool> running(true);


static size_t* _trSpace(size_t* size)
{
  static size_t trOffset[TransitionId::NumberOf];

  size_t sz = 0;
  for (unsigned tr = 0; tr < TransitionId::NumberOf; ++tr)
  {
    trOffset[tr] = sz;
    sz += sizeof(EbDgram) + (3 + tr) * sizeof(uint64_t); // 3 = min payload size
  }
  *size = roundUpSize(sz);

  return trOffset;
}

int server(const std::string& ifAddr,
           std::string&       srvPort,
           unsigned           id,
           unsigned           numClients,
           unsigned           numBuffers)
{
  const unsigned verbose(1);
  size_t         trSize;
  size_t*        trOffset = _trSpace(&trSize);
  EbLfServer*    svr      = new EbLfServer(verbose);
  int            rc;

  if ( (rc = svr->listen(ifAddr, srvPort, numClients)) )
  {
    fprintf(stderr, "%s:\n  Failed to initialize EbLfServer\n",
            __PRETTY_FUNCTION__);
    return rc;
  }

  std::vector<EbLfSvrLink*> links(numClients);
  std::vector<void*>        regions(numClients);
  std::vector<size_t>       bufSize(numClients);
  for (unsigned i = 0; i < links.size(); ++i)
  {
    const unsigned tmo(120000);
    if ( (rc = svr->connect(&links[i], tmo)) )
    {
      fprintf(stderr, "Error connecting to EbLfClient[%d]\n", i);
      return rc;
    }
    size_t regSize;
    if ( (rc = links[i]->prepare(id, &regSize, "Client")) < 0)
    {
      fprintf(stderr, "Failed to prepare link[%d]\n", i);
      return rc;
    }
    regions[i] = allocRegion(regSize);
    if (!regions[i])
    {
      fprintf(stderr, "No memory found for region %d of size %zd\n",
              i, regSize);
      return -1;
    }
    if ( (rc = links[i]->setupMr(regions[i], regSize, "Client")) )
    {
      fprintf(stderr, "Failed to set up MemoryRegion %d\n", i);
      return rc;
    }

    bufSize[i] = (regSize - trSize) / numBuffers;

    printf("EbLfClient[%d] (ID %d) connected with buffer size %zd\n",
           i, links[i]->id(), bufSize[i]);
  }

  while (running)
  {
    uint64_t  data;
    const int tmo = 5000;               // milliseconds
    if (svr->pend(&data, tmo) < 0)  continue;

    unsigned     flg = ImmData::flg(data);
    unsigned     src = ImmData::src(data);
    unsigned     idx = ImmData::idx(data);
    EbLfSvrLink* lnk = links[src];
    size_t       ofs = (ImmData::buf(flg) == ImmData::Buffer)
                     ? trSize + idx * bufSize[src]
                     : trOffset[idx];
    if ( (rc = links[src]->postCompRecv()) )
    {
      fprintf(stderr, "Failed to post CQ buffers: %d\n", rc);
    }

    void*       buf = lnk->lclAdx(ofs);
    uint64_t*   b   = (uint64_t*)buf;
    const char* k   = (ImmData::buf(flg) == ImmData::Buffer) ? "l1a" : "tr";
    printf("Rcvd buf %p, data %08lx [%2d %2d %2d]: cnt %2ld %3s %2ld c %2ld\n",
           buf, data, flg, src, idx, b[0], k, b[1], b[2]);
  }

  for (unsigned i = 0; i < links.size(); ++i)
  {
    free(regions[i]);
    svr->disconnect(links[i]);
  }
  delete svr;

  return 0;
}

int client(std::vector<std::string>& svrAddrs,
           std::vector<std::string>& svrPorts,
           unsigned                  id,
           unsigned                  numBuffers,
           size_t                    bufSize,
           unsigned                  iters,
           unsigned                  numEvents)
{
  static const int trId[TransitionId::NumberOf] =
    { TransitionId::ClearReadout,
      TransitionId::Reset,
      TransitionId::Configure,       TransitionId::Unconfigure,
      TransitionId::Enable,          TransitionId::Disable,
      TransitionId::L1Accept };
  const unsigned verbose(1);
  size_t         trSize;
  size_t*        trOffset = _trSpace(&trSize);
  size_t         regSize  = trSize + roundUpSize(numBuffers * bufSize);
  void*          region   = allocRegion(regSize);
  EbLfClient*    clt      = new EbLfClient(verbose);
  const unsigned tmo(120000);
  int            rc;

  // Adjust for the region size round up
  bufSize = (regSize - trSize) / numBuffers;

  std::vector<EbLfCltLink*> links(svrAddrs.size());
  for (unsigned i = 0; i < links.size(); ++i)
  {
    const char* svrAddr = svrAddrs[i].c_str();
    const char* svrPort = svrPorts[i].c_str();

    if ( (rc = clt->connect(&links[i], svrAddr, svrPort, tmo)) )
    {
      fprintf(stderr, "Error connecting to EbLfServer[%d]\n", i);
      return rc;
    }
    if ( (rc = links[i]->prepare(id, region, regSize, "Server")) < 0)
    {
      fprintf(stderr, "Failed to prepare link[%d]\n", i);
      return rc;
    }

    printf("EbLfServer[%d] (ID %d) connected for buffer size %zd\n",
           i, links[i]->id(), bufSize);
  }

  unsigned cnt = 0;
  unsigned c   = 0;
  while (cnt++ < iters)
  {
    void* buf;
    for (int i = 0; i < TransitionId::NumberOf - 1; i += 2)
    {
      int    tr     = trId[i];
      size_t offset = trOffset[tr];
      buf = (char*)region + offset;
      uint64_t* b = (uint64_t*)buf;
      *b++ = cnt;
      *b++ = tr;
      *b++ = c++;
      size_t size = (char*)b - (char*)buf;

      for (unsigned dst = 0; dst < links.size(); ++dst)
      {
        uint64_t data = ImmData::value(ImmData::Transition | ImmData::Response, id, tr);
        unsigned flg =  ImmData::flg(data);
        unsigned src  = ImmData::src(data);
        unsigned idx  = ImmData::idx(data);

        printf("Post buf %p, data %08lx [%2d %2d %2d]: cnt %2d tr  %2d c %2d - sz %zd to 0x%lx\n",
               buf, data, flg, src, idx, cnt, tr, c - 1, size, links[dst]->rmtAdx(offset));

        if ((rc = links[dst]->post(buf, size, offset, data)))
          fprintf(stderr, "Error %d posting transition %d to %d (ID %d)\n",
                  rc, tr, dst, links[dst]->id());
      }
    }

    size_t   offset = trSize;
    buf = (char*)region + offset;
    unsigned idx    = 0;
    for (unsigned l1a = 0; l1a < numEvents; ++l1a)
    {
      uint64_t* b = (uint64_t*)buf;
      *b++ = cnt;
      *b++ = l1a;
      *b++ = c++;
      size_t size = (char*)b - (char*)buf;

      for (unsigned dst = 0; dst < links.size(); ++dst)
      {
        int      rc;
        uint64_t data = ImmData::value(ImmData::Buffer | ImmData::Response, id, idx);
        unsigned flg =  ImmData::flg(data);
        unsigned src  = ImmData::src(data);
        unsigned idx  = ImmData::idx(data);

        printf("Post buf %p, data %08lx [%2d %2d %2d]: cnt %2d l1a %2d c %2d - sz %zd to 0x%lx\n",
               buf, data, flg, src, idx, cnt, l1a, c - 1, size, links[dst]->rmtAdx(offset));

        if ((rc = links[dst]->post(buf, size, offset, data)))
          fprintf(stderr, "Error %d posting L1Accept to %d (ID %d)\n",
                  rc, dst, links[dst]->id());
      }

      if (++idx < numBuffers)
      {
        offset += bufSize;
        buf     = (char*)buf + bufSize;
      }
      else
      {
        offset = trSize;
        buf    = (char*)region + offset;
        idx    = 0;
      }
    }

    for (int i = TransitionId::NumberOf - 2; i >= 0; i -= 2)
    {
      int    tr     = trId[i];
      size_t offset = trOffset[tr];
      buf = (char*)region + offset;
      uint64_t* b = (uint64_t*)buf;
      *b++ = cnt;
      *b++ = tr;
      *b++ = c++;
      size_t size = (char*)b - (char*)buf;

      for (unsigned dst = 0; dst < links.size(); ++dst)
      {
        uint64_t data = ImmData::value(ImmData::Transition | ImmData::Response, id, tr);
        unsigned flg  = ImmData::flg(data);
        unsigned src  = ImmData::src(data);
        unsigned idx  = ImmData::idx(data);

        printf("Post buf %p, data %08lx [%2d %2d %2d]: cnt %2d tr  %2d c %2d - sz %zd to 0x%lx\n",
               buf, data, flg, src, idx, cnt, tr, c - 1, size, links[dst]->rmtAdx(offset));

        if ((rc = links[dst]->post(buf, size, offset, data)))
          fprintf(stderr, "Error %d posting transition %d to %d (ID %d)\n",
                  rc, tr, dst, links[dst]->id());
      }
    }
    // Need a handshake with the other side here to avoid stepping on data that
    // hasn't been handled yet, but for this purpose we kludge it with a sleep
    usleep(500);                    // May need adjustment for other conditions
  }

  for (unsigned dst = 0; dst < links.size(); ++dst)
  {
    clt->disconnect(links[dst]);
  }
  delete clt;

  return 0;
}


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
  fprintf(stderr, "  %s [OPTIONS]\n", name);

  fprintf(stderr, "\nWhere:\n"
                  "  <spec>, below, has the form '<id>:<addr>:<port>', with\n");
  fprintf(stderr, "  <id> typically in the range 0 - 63.\n");

  fprintf(stderr, "\nOptions:\n");

  fprintf(stderr, " %-20s %s (default: %s)\n",      "-A <interface_addr>",
          "IP address of the interface to use",     "libfabric's 'best' choice");
  fprintf(stderr, " %-20s %s (default: %s)\n",      "-S \"<spec> [<spec> [...]]\"",
          "Servers to connect to",                  "None, must be provided for clients");
  fprintf(stderr, " %-20s %s (port: %d)\n",         "-P <port>",
          "Base port number",                       port_base);

  fprintf(stderr, " %-20s %s (default: %d)\n",      "-i <ID>",
          "Unique ID of this client (0 - 63)",      default_id);
  fprintf(stderr, " %-20s %s (default: %d)\n",      "-n <iters>",
          "Number of times to iterate",             default_iters);
  fprintf(stderr, " %-20s %s (default: %s)\n",      "-c <clients>",
          "Number of clients",                      "None, must be provided for servers");
  fprintf(stderr, " %-20s %s (default: %d)\n",      "-b <buffers>",
          "Number of buffers",                      default_buf_cnt);
  fprintf(stderr, " %-20s %s (default: %d)\n",      "-s <buffer size>",
          "Max buffer size (ignored by server)",    default_buf_size);

  fprintf(stderr, " %-20s %s\n", "-h", "display this help output");
}

static int parseSpec(char*                     spec,
                     unsigned                  maxId,
                     unsigned                  portMin,
                     unsigned                  portMax,
                     std::vector<std::string>& addrs,
                     std::vector<std::string>& ports,
                     uint64_t*                 bits)
{
  do
  {
    char* target = strsep(&spec, ",");
    if (!*target)  continue;
    char* colon1 = strchr(target, ':');
    char* colon2 = strrchr(target, ':');
    if (!colon1 || (colon1 == colon2))
    {
      fprintf(stderr, "Input '%s' is not of the form <ID>:<IP>:<port>\n", target);
      return 1;
    }
    unsigned id   = atoi(target);
    unsigned port = atoi(&colon2[1]);
    if (id > maxId)
    {
      fprintf(stderr, "ID %d is out of range 0 - %d\n", id, maxId);
      return 1;
    }
    if (port < portMax - portMin + 1)  port += portMin;
    if ((port < portMin) || (port > portMax))
    {
      fprintf(stderr, "Port %d is out of range %d - %d\n",
              port, portMin, portMax);
      return 1;
    }
    *bits |= 1ul << id;
    addrs.push_back(std::string(&colon1[1]).substr(0, colon2 - &colon1[1]));
    ports.push_back(std::string(std::to_string(port)));
  }
  while (spec);

  return 0;
}

int main(int argc, char **argv)
{
  int         op, rc   = 0;
  unsigned    id       = default_id;
  std::string ifAddr   = { };
  unsigned    portBase = port_base;
  unsigned    iters    = default_iters;
  unsigned    nEvents  = default_num_evts;
  unsigned    nClients = 0;
  unsigned    nBuffers = default_buf_cnt;
  size_t      bufSize  = default_buf_size;
  char*       spec     = nullptr;

  while ((op = getopt(argc, argv, "h?A:S:P:i:n:N:c:b:s:")) != -1)
  {
    switch (op)
    {
      case 'A':  ifAddr   = optarg;        break;
      case 'S':  spec     = optarg;        break;
      case 'P':  portBase = atoi(optarg);  break;
      case 'i':  id       = atoi(optarg);  break;
      case 'n':  iters    = atoi(optarg);  break;
      case 'N':  nEvents  = atoi(optarg);  break;
      case 'c':  nClients = atoi(optarg);  break;
      case 'b':  nBuffers = atoi(optarg);  break;
      case 's':  bufSize  = atoi(optarg);  break;
      case '?':
      case 'h':
      default:
        usage(argv[0], (char*)"Test of EbLf links");
        return 1;
    }
  }

  if (id >= max_clients)
  {
    fprintf(stderr, "Client ID %d is out of range 0 - %d\n", id, max_clients - 1);
    return 1;
  }

  if ((portBase < port_base) || (portBase > port_base + max_clients))
  {
    fprintf(stderr, "Server port %d is out of range %d - %d\n",
            portBase, port_base, port_base + max_servers);
    return 1;
  }
  std::string srvPort(std::to_string(portBase));

  std::vector<std::string> cltAddrs;
  std::vector<std::string> cltPorts;
  uint64_t clients = 0;
  if (spec)
  {
    int rc = parseSpec(spec,
                       max_servers - 1,
                       port_base,
                       port_base + max_servers - 1,
                       cltAddrs,
                       cltPorts,
                       &clients);
    if (rc)  return rc;
  }

  ::signal( SIGINT, sigHandler );

  if (cltAddrs.size() == 0)
  {
    if (nClients == 0)
    {
      fprintf(stderr, "Number of clients (-c) must be provided\n");
      return 1;
    }

    rc = server(ifAddr, srvPort, id, nClients, nBuffers);
  }
  else
  {
    if (spec == nullptr)
    {
      fprintf(stderr, "Server spec(s) (-S) must be provided\n");
      return 1;
    }

    rc = client(cltAddrs, cltPorts, id, nBuffers, bufSize, iters, nEvents);
  }

  return rc;
}
