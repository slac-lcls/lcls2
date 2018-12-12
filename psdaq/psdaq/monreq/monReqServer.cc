#include "psdaq/monreq/XtcMonitorServer.hh"

#include "psdaq/eb/eb.hh"
#include "psdaq/eb/EbAppBase.hh"
#include "psdaq/eb/EbEvent.hh"

#include "psdaq/eb/EbLfClient.hh"

#include "psdaq/eb/utilities.hh"
#include "psdaq/eb/StatsMonitor.hh"

#include "psdaq/service/Fifo.hh"
#include "psdaq/service/GenericPool.hh"
#include "psdaq/service/Collection.hh"
#include "xtcdata/xtc/Dgram.hh"

#include <errno.h>
#include <unistd.h>                     // For getopt()
#include <string.h>
#include <vector>
#include <bitset>
#include <iostream>

static const unsigned rtMon_period         = 1;  // Seconds
static const unsigned default_id           = 0;  // Builder's ID (< 64)
static const unsigned epoch_duration       = 8;  // Revisit: 1 per xferBuffer
static const unsigned numberof_xferBuffers = 8;  // Revisit: Value; corresponds to tstEbContributor:maxEvents
static const unsigned sizeof_buffers       = 1024; // Revisit

static       unsigned lverbose             = 0;

using namespace XtcData;
using namespace Pds::Eb;
using namespace Pds::MonReq;
using namespace Pds;

namespace Pds {
  class MyXtcMonitorServer : public XtcMonitorServer {
  public:
    MyXtcMonitorServer(const char*               tag,
                       unsigned                  sizeofBuffers,
                       unsigned                  numberofEvBuffers,
                       unsigned                  numberofEvQueues,
                       std::vector<std::string>& addrs,
                       std::vector<std::string>& ports,
                       unsigned                  id) :
      XtcMonitorServer(tag,
                       sizeofBuffers,
                       numberofEvBuffers,
                       numberofEvQueues),
      _sizeofBuffers(sizeofBuffers),
      _iTeb(0),
      _mrqTransport(new EbLfClient()),
      _mrqLinks(addrs.size()),
      _bufFreeList(numberofEvBuffers),
      _id(id)
    {
      for (unsigned i = 0; i < addrs.size(); ++i)
      {
        const char*    addr = addrs[i].c_str();
        const char*    port = ports[i].c_str();
        EbLfLink*      link;
        const unsigned tmo(120000);     // Milliseconds
        if (_mrqTransport->connect(addr, port, tmo, &link))
        {
          fprintf(stderr, "%s:\n  Error connecting to Mon EbLfServer at %s:%s\n",
                  __func__, addr, port);
          abort();
        }
        if (link->preparePoster(i, id, lverbose))
        {
          fprintf(stderr, "%s:\n  Failed to prepare Mon link to %s:%s\n",
                  __func__, addr, port);
          abort();
        }
        _mrqLinks[i] = link;

        printf("%s: EbLfServer ID %d connected\n", __func__, link->id());
      }

      for (unsigned i = 0; i < numberofEvBuffers; ++i)
        if (!_bufFreeList.push(i))
          fprintf(stderr, "%s:\n  _bufFreeList.push(%d) failed\n", __func__, i);
    }
    virtual ~MyXtcMonitorServer()
    {
      if (_mrqTransport)  delete _mrqTransport;
    }
  public:
    void startup(bool dist)
    {
      distribute(dist);

      _init();
    }
    void shutdown()
    {
      if (_mrqTransport)
      {
        for (std::vector<EbLfLink*>::iterator it  = _mrqLinks.begin();
                                              it != _mrqLinks.end(); ++it)
        {
          _mrqTransport->shutdown(*it);
        }
        _mrqLinks.clear();
      }
    }

  private:
    virtual void _copyDatagram(Dgram* dg, char* buf)
    {
      //printf("_copyDatagram @ %p to %p: pid = %014lx\n",
      //       dg, buf, dg->seq.pulseId().value());

      Dgram* odg = new((void*)buf) Dgram(*dg);

      // The dg payload is a directory of contributions to the built event.
      // Iterate over the directory and construct, in shared memory, the event
      // datagram (odg) from the contribution XTCs
      const Dgram** const  last = (const Dgram**)dg->xtc.next();
      const Dgram*  const* ctrb = (const Dgram**)dg->xtc.payload();
      do
      {
        const Dgram* idg = *ctrb;

        buf = (char*)odg->xtc.alloc(idg->xtc.extent);

        if (sizeof(*odg) + odg->xtc.sizeofPayload() > _sizeofBuffers)
        {
          fprintf(stderr, "%s:\n  Datagram is too large (%zd) for buffer of size %d\n",
                  __PRETTY_FUNCTION__, sizeof(*odg) + odg->xtc.sizeofPayload(), _sizeofBuffers);
          abort();            // The memcpy would blow by the buffer size limit
        }

        memcpy(buf, &idg->xtc, idg->xtc.extent);
      }
      while (++ctrb != last);
    }

    virtual void _deleteDatagram(Dgram* dg)
    {
      //printf("_deleteDatagram @ %p\n", dg);

      unsigned idx = *(unsigned*)dg->xtc.next();
      if (!_bufFreeList.push(idx))
        printf("_bufFreeList.push(%d) failed, count = %zd\n", idx, _bufFreeList.count());
      //printf("_deleteDatagram: _bufFreeList.push(%d), count = %zd, log = %08x, phy = %08x\n",
      //       idx, _bufFreeList.count(), dg->xtc.src.value(), dg->xtc.src.phy());

      Pool::free((void*)dg);
    }

    virtual void _requestDatagram()
    {
      //printf("_requestDatagram\n");

      if (_bufFreeList.empty())
      {
        fprintf(stderr, "%s:\n  No free buffers available\n", __PRETTY_FUNCTION__);
        return;
      }

      uint32_t data = _bufFreeList.pop();
      //printf("_requestDatagram: _bufFreeList.pop(): %08x, count = %zd\n", data, _bufFreeList.count());

      int rc = -1;
      for (unsigned i = 0; i < _mrqLinks.size(); ++i)
      {
        // Round robin through Trigger Event Builders
        unsigned iTeb = _iTeb++;
        if (_iTeb == _mrqLinks.size())  _iTeb = 0;

        EbLfLink* link = _mrqLinks[iTeb];

        data = ImmData::value(ImmData::Buffer, _id /*link->index()*/, data);

        rc = link->post(nullptr, 0, data);

        //printf("_requestDatagram: Post %d EB[iTeb = %d], value = %08x, rc = %d\n",
        //       i, iTeb, data, rc);

        if (rc == 0)  break;            // Break if message was delivered
      }
      if (rc)
      {
        fprintf(stderr, "%s:\n  Unable to post request to any TEB\n", __PRETTY_FUNCTION__);
        // Revisit: Is this fatal or ignorable?
      }
    }

  private:
    unsigned               _sizeofBuffers;
    unsigned               _nTeb;
    unsigned               _iTeb;
    EbLfClient*            _mrqTransport;
    std::vector<EbLfLink*> _mrqLinks;
    Fifo<unsigned>         _bufFreeList;
    unsigned               _id;
  };

  static size_t _calcBufSize(size_t sizeofBuffers, uint64_t contributors)
  {
    // sizeofBuffers is the max size of a built event in shared memory
    size_t sz = sizeofBuffers / std::bitset<64>(contributors).count();
    return sz;
  }

  class MebApp : public EbAppBase
  {
  public:
    MebApp(const char*        ifAddr,
           const std::string& port,
           unsigned           id,
           unsigned           sizeofEvBuffers,
           unsigned           sizeofTrBuffers,
           unsigned           numberofEvBuffers,
           uint64_t           contributors,
           StatsMonitor&      smon) :
      EbAppBase(ifAddr, port,           // EbLfServer params to Contributor link
                id,                     // This monitor's ID
                epoch_duration,         // Time duration of a buffer
                numberofEvBuffers,      // Max # of Event buffers
                1,                      // # of dgs per buffer
                _calcBufSize(sizeofEvBuffers, contributors), // Size per dg
                sizeofTrBuffers,        // Size per non-event
                contributors),          // Same value as TEB's
      _eventCount(0),
      _apps      (nullptr),
      _pool      (sizeof(Dgram)                                          +
                  std::bitset<64>(contributors).count() * sizeof(Dgram*) +
                  sizeof(unsigned),
                  numberof_xferBuffers) // Revisit: numberof_xferBuffers
    {
      smon.registerIt("MEB.EvtRt",  _eventCount,      StatsMonitor::RATE);
      smon.registerIt("MEB.EvtCt",  _eventCount,      StatsMonitor::SCALAR);
      smon.registerIt("MEB.EpAlCt",  epochAllocCnt(), StatsMonitor::SCALAR);
      smon.registerIt("MEB.EpFrCt",  epochFreeCnt(),  StatsMonitor::SCALAR);
      smon.registerIt("MEB.EvAlCt",  eventAllocCnt(), StatsMonitor::SCALAR);
      smon.registerIt("MEB.EvFrCt",  eventFreeCnt(),  StatsMonitor::SCALAR);
      smon.registerIt("MEB.RxPdg",   rxPending(),     StatsMonitor::SCALAR);
    }
    virtual ~MebApp()
    {
    }
  public:
    void process(MyXtcMonitorServer* apps, bool dist)
    {
      _apps = apps;

      //pinThread(task()->parameters().taskID(), lcore2);
      //pinThread(pthread_self(),                lcore2);

      //start();                              // Start the event timeout timer

      //pinThread(pthread_self(),                lcore1);

      while (true)
      {
        EbAppBase::process();
      }

      //cancel();                         // Stop the event timeout timer

      EbAppBase::shutdown();
    }
    virtual void process(EbEvent* event)
    {
      if (lverbose > 3)
      {
        static unsigned cnt = 0;
        printf("MebApp::process event dump:\n");
        event->dump(++cnt);
      }
      ++_eventCount;

      // Create a Dgram with a payload that is a directory of contribution
      // Dgrams to the built event.  Reserve space at end for the buffer's index
      size_t   sz     = (event->end() - event->begin()) * sizeof(*(event->begin()));
      unsigned idx    = ImmData::idx(event->parameter());
      void*    buffer = _pool.alloc(sizeof(Dgram) + sz + sizeof(idx));
      if (!buffer)
      {
        fprintf(stderr, "%s:\n  Dgram pool allocation failed:\n", __PRETTY_FUNCTION__);
        _pool.dump();
        abort();
      }
      Dgram*  dg  = new(buffer) Dgram(*(event->creator()));
      Dgram** buf = (Dgram**)dg->xtc.alloc(sz);
      memcpy(buf, event->begin(), sz);
      *(unsigned*)dg->xtc.next() = idx; // Pass buffer's index to _deleteDatagram()

      if (lverbose > 2)
      {
        printf("MEB processed              event[%4ld]    @ "
               "%16p, pid %014lx, sz %4zd, buf # %2d\n",
               _eventCount, dg, dg->seq.pulseId().value(),
               sizeof(*dg) + dg->xtc.sizeofPayload(), idx);
      }

      if (_apps->events(dg) == XtcMonitorServer::Handled)
      {
        Pool::free((void*)dg);
      }
    }
  private:
    uint64_t          _eventCount;
    XtcMonitorServer* _apps;
    GenericPool       _pool;
  };
};

using namespace Pds;


void usage(char* progname)
{
  printf("\n<MRQ_spec> has the form '<id>:<addr>:<port>'\n");
  printf("<id> must be in the range 0 - %d.\n", MAX_MEBS - 1);
  printf("Low numbered <port> values are treated as offsets into the following range:\n");
  printf("  Mon requests: %d - %d\n", MRQ_PORT_BASE, MRQ_PORT_BASE + MAX_MEBS - 1);

  printf("Usage: %s -C <collection server>"
                   "-p <platform> "
                   "[-P <partition>] "
                   "-n <numb shm buffers> "
                   "-s <shm buffer size> "
                  "[-q <# event queues>] "
                  "[-t <tag name>] "
                  "[-d] "
                  "[-A <interface addr>] "
                  "[-E <MEB port>] "
                  "[-i <ID>] "
                   "-c <Contributors (DRPs)> "
                  "[-Z <Run-time mon host>] "
                  "[-R <Run-time mon port>] "
                  "[-m <Run-time mon publishing period>] "
                  "[-V] " // Run-time mon verbosity
                  "[-v] "
                  "[-h] "
                  "<MRQ_spec> [MRQ_spec [...]]\n", progname);
}

static
void joinCollection(std::string&              server,
                    unsigned                  partition,
                    unsigned                  portBase,
                    uint64_t&                 contributors,
                    std::vector<std::string>& addrs,
                    std::vector<std::string>& ports,
                    unsigned&                 mebId)
{
  Collection collection(server, partition, "meb");
  collection.connect();
  std::cout << "cmstate:\n" << collection.cmstate.dump(4) << std::endl;

  std::string id = std::to_string(collection.id());
  mebId = collection.cmstate["meb"][id]["meb_id"];
  //std::cout << "MEB: ID " << mebId << std::endl;

  for (auto it : collection.cmstate["drp"].items())
  {
    unsigned ctrbId = it.value()["drp_id"];
    //std::cout << "DRP: ID " << ctrbId << std::endl;
    contributors |= 1ul << ctrbId;
  }

  for (auto it : collection.cmstate["teb"].items())
  {
    unsigned    tebId  = it.value()["teb_id"];
    std::string address = it.value()["connect_info"]["infiniband"];
    //std::cout << "TEB: ID " << tebId << "  " << address << std::endl;
    addrs.push_back(address);
    ports.push_back(std::string(std::to_string(portBase + tebId)));
    //printf("MRQ Clt[%d] port = %d\n", tebId, portBase + tebId);
  }
}

int main(int argc, char** argv)
{
  const unsigned NO_PLATFORM     = unsigned(-1UL);
  unsigned       platform        = NO_PLATFORM;
  const char*    tag             = 0;
  unsigned       numberofBuffers = numberof_xferBuffers;
  unsigned       sizeofEvBuffers = sizeof_buffers;
  unsigned       sizeofTrBuffers = sizeof_buffers;
  unsigned       nevqueues       = 1;
  bool           ldist           = false;
  unsigned       id              = default_id;
  char*          ifAddr          = nullptr;
  unsigned       mebPortNo       = 0;   // Port served to contributors
  uint64_t       contributors    = 0;
  std::string    partition        (PARTITION);
  const char*    rtMonHost       = RTMON_HOST;
  unsigned       rtMonPort       = RTMON_PORT_BASE;
  unsigned       rtMonPeriod     = rtMon_period;
  unsigned       rtMonVerbose    = 0;
  std::string    collSvr          (COLL_HOST);

  int c;
  while ((c = getopt(argc, argv, "p:n:P:s:q:t:dA:E:i:c:Z:R:m:C:Vvh")) != -1)
  {
    errno = 0;
    char* endPtr;
    switch (c) {
      case 'p':
        platform = strtoul(optarg, &endPtr, 0);
        if (errno != 0 || endPtr == optarg) platform = NO_PLATFORM;
        break;
      case 'n':
        sscanf(optarg, "%d", &numberofBuffers);
        break;
      case 'P':
        partition = std::string(optarg);
        break;
      case 't':
        tag = optarg;
        break;
      case 'q':
        nevqueues = strtoul(optarg, NULL, 0);
        break;
      case 's':
        sizeofEvBuffers = (unsigned) strtoul(optarg, NULL, 0);
        break;
      case 'd':
        ldist = true;
        break;
      case 'A':  ifAddr       = optarg;                       break;
      case 'E':  mebPortNo    = atoi(optarg);                 break;
      case 'i':  id           = atoi(optarg);                 break;
      case 'c':  contributors = strtoul(optarg, nullptr, 0);  break;
      case 'Z':  rtMonHost    = optarg;                       break;
      case 'R':  rtMonPort    = atoi(optarg);                 break;
      case 'm':  rtMonPeriod  = atoi(optarg);                 break;
      case 'C':  collSvr      = optarg;                       break;
      case 'V':  ++rtMonVerbose;                              break;
      case 'v':  ++lverbose;                                  break;
      case 'h':                         // help
        usage(argv[0]);
        return 0;
        break;
      default:
        printf("Unrecogized parameter\n");
        usage(argv[0]);
        return 1;
    }
  }

  if (numberofBuffers < numberof_xferBuffers) numberofBuffers = numberof_xferBuffers;

  if (!tag) tag=partition.c_str();
  printf("Partition Tag: '%s'\n", tag);

  const unsigned numPorts    = MAX_DRPS + MAX_TEBS + MAX_MEBS + MAX_MEBS;
  const unsigned mrqPortBase = MRQ_PORT_BASE + numPorts * platform;
  const unsigned mebPortBase = MEB_PORT_BASE + numPorts * platform;

  std::vector<std::string> mrqAddrs;
  std::vector<std::string> mrqPorts;
  if (optind < argc)
  {
    do
    {
      char* contributor = argv[optind];
      char* colon1      = strchr(contributor, ':');
      char* colon2      = strrchr(contributor, ':');
      if (!colon1 || (colon1 == colon2))
      {
        fprintf(stderr, "MRQ input '%s' is not of the form <ID>:<IP>:<port>\n", contributor);
        return 1;
      }
      unsigned port = atoi(&colon2[1]);
      if (port < MAX_MEBS)  port += mrqPortBase;
      if ((port < mrqPortBase) || (port >= mrqPortBase + MAX_MEBS))
      {
        fprintf(stderr, "MRQ client port %d is out of range %d - %d\n",
                port, mrqPortBase, mrqPortBase + MAX_MEBS);
        return 1;
      }
      mrqAddrs.push_back(std::string(&colon1[1]).substr(0, colon2 - &colon1[1]));
      mrqPorts.push_back(std::string(std::to_string(port)));
    }
    while (++optind < argc);
  }
  else
  {
    joinCollection(collSvr, platform, mrqPortBase, contributors, mrqAddrs, mrqPorts, id);
  }
  if (id >= MAX_MEBS)
  {
    fprintf(stderr, "MEB ID %d is out of range 0 - %d\n", id, MAX_MEBS - 1);
    return 1;
  }
  if ((mrqAddrs.size() == 0) || (mrqPorts.size() == 0))
  {
    fprintf(stderr, "Missing required TEB request address(es)\n");
    return 1;
  }

  if  (mebPortNo < MAX_MEBS)  mebPortNo += mebPortBase;
  if ((mebPortNo < mebPortBase) || (mebPortNo >= mebPortBase + MAX_MEBS))
  {
    fprintf(stderr, "MEB Server port %d is out of range %d - %d\n",
            mebPortNo, mebPortBase, mebPortBase + MAX_MEBS);
    return 1;
  }
  std::string mebPort(std::to_string(mebPortNo + id));
  //printf("MEB Srv port = %s\n", mebPort.c_str());

  if (!numberofBuffers || !sizeofEvBuffers || platform == NO_PLATFORM || !contributors) {
    fprintf(stderr, "Missing parameters!\n");
    usage(argv[0]);
    return 1;
  }

  EbAppBase::lverbose = lverbose;
  EventBuilder::lverbose = lverbose;

  StatsMonitor*       smon = new StatsMonitor(rtMonHost,
                                              rtMonPort,
                                              platform,
                                              partition,
                                              rtMonPeriod,
                                              rtMonVerbose);

  MebApp*             meb  = new MebApp(ifAddr, mebPort,
                                        id,
                                        sizeofEvBuffers,
                                        sizeofTrBuffers,
                                        numberofBuffers,
                                        contributors,
                                        *smon);

  MyXtcMonitorServer* apps = new MyXtcMonitorServer(tag,
                                                    sizeofEvBuffers,
                                                    numberofBuffers,
                                                    nevqueues,
                                                    mrqAddrs, mrqPorts,
                                                    id);

  // Wait a bit to allow other components of the system to establish connections
  sleep(1);

  apps->startup(ldist);

  meb->process(apps, ldist);

  meb->shutdown();
  apps->shutdown();

  delete apps;
  delete meb;
  delete smon;

  return 0;
}
