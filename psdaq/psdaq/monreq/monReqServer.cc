#include "psalg/shmem/XtcMonitorServer.hh"

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

#include <signal.h>
#include <errno.h>
#include <unistd.h>                     // For getopt()
#include <string.h>
#include <vector>
#include <bitset>
#include <iostream>
#include <atomic>

static const int      core_0               = 10; // devXXX: 10, devXX:  7, accXX:  9
static const int      core_1               = 11; // devXXX: 11, devXX: 19, accXX: 21
static const unsigned rtMon_period         = 1;  // Seconds
static const unsigned epoch_duration       = 8;  // Revisit: 1 per xferBuffer
static const unsigned numberof_xferBuffers = 8;  // Revisit: Value; corresponds to tstEbContributor:maxEvents
static const unsigned sizeof_buffers       = 1024; // Revisit

using namespace XtcData;
using namespace Pds::Eb;
using namespace psalg::shmem;
using namespace Pds;

static struct sigaction      lIntAction;
static volatile sig_atomic_t lRunning = 1;

void sigHandler( int signal )
{
  static unsigned callCount(0);

  if (callCount == 0)
  {
    printf("\nShutting down\n");

    lRunning = 0;
  }

  if (callCount++)
  {
    fprintf(stderr, "Aborting on 2nd ^C...\n");
    ::abort();
  }
}


namespace Pds {
  class MyXtcMonitorServer : public XtcMonitorServer {
  public:
    MyXtcMonitorServer(const char*     tag,
                       unsigned        sizeofBuffers,
                       unsigned        numberofEvQueues,
                       const EbParams& prms) :
      XtcMonitorServer(tag,
                       sizeofBuffers,
                       prms.maxBuffers,
                       numberofEvQueues),
      _sizeofBuffers(sizeofBuffers),
      _iTeb(0),
      _mrqTransport(prms.verbose),
      _mrqLinks(),
      _bufFreeList(prms.maxBuffers),
      _id(-1u)
    {
    }
    int connect(const EbParams& prms)
    {
      _id = prms.id;
      _mrqLinks.resize(prms.addrs.size());

      for (unsigned i = 0; i < prms.addrs.size(); ++i)
      {
        int            rc;
        const char*    addr = prms.addrs[i].c_str();
        const char*    port = prms.ports[i].c_str();
        EbLfLink*      link;
        const unsigned tmo(120000);     // Milliseconds
        if ( (rc = _mrqTransport.connect(addr, port, tmo, &link)) )
        {
          fprintf(stderr, "%s:\n  Error connecting to Monitor EbLfServer at %s:%s\n",
                  __func__, addr, port);
          return rc;
        }
        if ( (rc = link->preparePoster(_id)) )
        {
          fprintf(stderr, "%s:\n  Failed to prepare Monitor link to %s:%s\n",
                  __func__, addr, port);
          return rc;
        }
        _mrqLinks[link->id()] = link;

        printf("Outbound link with TEB ID %d connected\n", link->id());
      }

      for (unsigned i = 0; i < prms.maxBuffers; ++i)
        if (!_bufFreeList.push(i))
          fprintf(stderr, "%s:\n  _bufFreeList.push(%d) failed\n", __func__, i);

      _init();

      return 0;
    }
    virtual ~MyXtcMonitorServer()
    {
    }
  public:
    void shutdown()
    {
      for (auto it = _mrqLinks.begin(); it != _mrqLinks.end(); ++it)
      {
        _mrqTransport.shutdown(*it);
      }
      _mrqLinks.clear();
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
      //printf("_deleteDatagram: _bufFreeList.push(%d), count = %zd\n",
      //       idx, _bufFreeList.count());

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

        data = ImmData::value(ImmData::Buffer, _id, data);

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
    EbLfClient             _mrqTransport;
    std::vector<EbLfLink*> _mrqLinks;
    Fifo<unsigned>         _bufFreeList;
    unsigned               _id;
  };

  class Meb : public EbAppBase
  {
  public:
    Meb(const char*     tag,
        unsigned        sizeofEvBuffers,
        unsigned        nevqueues,
        bool            dist,
        const EbParams& prms,
        StatsMonitor&   smon) :
      EbAppBase  (prms),
      _apps      (tag, sizeofEvBuffers, nevqueues, prms),
      _pool      (nullptr),
      _eventCount(0),
      _verbose   (prms.verbose),
      _prms      (prms)
    {
      _apps.distribute(dist);

      smon.registerIt("MEB.EvtRt",  _eventCount,      StatsMonitor::RATE);
      smon.registerIt("MEB.EvtCt",  _eventCount,      StatsMonitor::SCALAR);
      smon.registerIt("MEB.EpAlCt",  epochAllocCnt(), StatsMonitor::SCALAR);
      smon.registerIt("MEB.EpFrCt",  epochFreeCnt(),  StatsMonitor::SCALAR);
      smon.registerIt("MEB.EvAlCt",  eventAllocCnt(), StatsMonitor::SCALAR);
      smon.registerIt("MEB.EvFrCt",  eventFreeCnt(),  StatsMonitor::SCALAR);
      smon.registerIt("MEB.RxPdg",   rxPending(),     StatsMonitor::SCALAR);
    }
    virtual ~Meb()
    {
    }
  public:
    int connect(const EbParams& prms)
    {
      unsigned entries = std::bitset<64>(prms.contributors).count();
      _pool = new GenericPool(sizeof(Dgram) + entries * sizeof(Dgram*) +
                              sizeof(unsigned), prms.maxBuffers);

      int rc;
      if ( (rc = EbAppBase::connect(prms)) )
        return rc;

      if ( (rc = _apps.connect(prms)) )
        return rc;

      return 0;
    }
    void run()
    {
      //pinThread(task()->parameters().taskID(), _prms.core[1]);
      //pinThread(pthread_self(),                _prms.core[1]);

      //start();                              // Start the event timeout timer

      //pinThread(pthread_self(),                _prms.core[0]);

      int rc;
      while (lRunning)
      {
        if ( (rc = EbAppBase::process()) )
        {
          if (rc != -FI_ETIMEDOUT )  break;
        }
      }

      //cancel();                         // Stop the event timeout timer

      _apps.shutdown();

      EbAppBase::shutdown();

      delete _pool;
      _pool = nullptr;
    }
    virtual void process(EbEvent* event)
    {
      if (_verbose > 2)
      {
        static unsigned cnt = 0;
        printf("Meb::process event dump:\n");
        event->dump(++cnt);
      }
      ++_eventCount;

      // Create a Dgram with a payload that is a directory of contribution
      // Dgrams to the built event.  Reserve space at end for the buffer's index
      size_t   sz     = (event->end() - event->begin()) * sizeof(*(event->begin()));
      unsigned idx    = ImmData::idx(event->parameter());
      void*    buffer = _pool->alloc(sizeof(Dgram) + sz + sizeof(idx));
      if (!buffer)
      {
        fprintf(stderr, "%s:\n  Dgram pool allocation failed:\n", __PRETTY_FUNCTION__);
        _pool->dump();
        abort();
      }
      Dgram*  dg  = new(buffer) Dgram(*(event->creator()));
      Dgram** buf = (Dgram**)dg->xtc.alloc(sz);
      memcpy(buf, event->begin(), sz);
      *(unsigned*)dg->xtc.next() = idx; // Pass buffer's index to _deleteDatagram()

      if (_verbose > 1)
      {
        uint64_t pid = dg->seq.pulseId().value();
        unsigned ctl = dg->seq.pulseId().control();
        size_t   sz  = sizeof(*dg) + dg->xtc.sizeofPayload();
        unsigned svc = dg->seq.service();
        printf("MEB processed              event[%4ld]    @ "
               "%16p, ctl %02x, pid %014lx, sz %4zd, %3s # %2d\n",
               _eventCount, dg, ctl, pid, sz,
               svc == TransitionId::L1Accept ? "buf" : "tr", idx);
      }

      if (_apps.events(dg) == XtcMonitorServer::Handled)
      {
        Pool::free((void*)dg);
      }
    }
  private:
    MyXtcMonitorServer _apps;
    GenericPool*       _pool;
    uint64_t           _eventCount;
    const unsigned     _verbose;
    const EbParams&    _prms;
  };
};


class MebApp : public CollectionApp
{
public:
  MebApp(const std::string& collSrv,
         const char*        tag,
         unsigned           sizeofEvBuffers,
         unsigned           nevqueues,
         bool               dist,
         EbParams&          prms,
         StatsMonitor&      smon);
public:                                 // For CollectionApp
  void handleAlloc(const json& msg) override;
  void handleConnect(const json& msg) override;
  void handleDisconnect(const json& msg); // override;
  void handleReset(const json& msg) override;
private:
  int  _parseConnectionParams(const json& msg);
  void _shutdown();
private:
  EbParams&     _prms;
  Meb           _meb;
  StatsMonitor& _smon;
  std::thread*  _appThread;
};

MebApp::MebApp(const std::string& collSrv,
               const char*        tag,
               unsigned           sizeofEvBuffers,
               unsigned           nevqueues,
               bool               dist,
               EbParams&          prms,
               StatsMonitor&      smon) :
  CollectionApp(collSrv, prms.partition, "meb"),
  _prms(prms),
  _meb(tag, sizeofEvBuffers, nevqueues, dist, prms, smon),
  _smon(smon),
  _appThread(nullptr)
{
}

void MebApp::handleAlloc(const json& msg)
{
  // Allow the default NIC choice to be overridden
  std::string nicIp = _prms.ifAddr.empty() ? getNicIp() : _prms.ifAddr;
  std::cout << "nic ip  " << nicIp << '\n';
  json body = {{getLevel(), {{"connect_info", {{"nic_ip", nicIp}}}}}};
  json answer = createMsg("alloc", msg["header"]["msg_id"], getId(), body);
  reply(answer);
}

void MebApp::handleConnect(const json &msg)
{
  int rc = _parseConnectionParams(msg["body"]);
  if (rc)
  {
    // Reply to collection with a failure message
    return;
  }

  rc = _meb.connect(_prms);
  if (rc)
  {
    // Reply to collection with a failure message
    return;
  }

  _smon.enable();

  lRunning = 1;

  _appThread = new std::thread(&Meb::run, std::ref(_meb));

  // Reply to collection with connect status
  json body   = json({});
  json answer = createMsg("connect", msg["header"]["msg_id"], getId(), body);
  reply(answer);
}

void MebApp::_shutdown()
{
  lRunning = 0;

  if (_appThread)
  {
    _appThread->join();
    delete _appThread;
    _appThread = nullptr;

    _smon.disable();
  }
}

void MebApp::handleDisconnect(const json &msg)
{
  _shutdown();

  // Reply to collection with connect status
  json body   = json({});
  json answer = createMsg("disconnect", msg["header"]["msg_id"], getId(), body);
  reply(answer);
}

void MebApp::handleReset(const json &msg)
{
  _shutdown();
}

int MebApp::_parseConnectionParams(const json& body)
{
  const unsigned numPorts    = MAX_DRPS + MAX_TEBS + MAX_TEBS + MAX_MEBS;
  const unsigned mrqPortBase = MRQ_PORT_BASE + numPorts * _prms.partition;
  const unsigned mebPortBase = MEB_PORT_BASE + numPorts * _prms.partition;

  std::string id = std::to_string(getId());
  _prms.id       = body["meb"][id]["meb_id"];
  if (_prms.id >= MAX_MEBS)
  {
    fprintf(stderr, "MEB ID %d is out of range 0 - %d\n", _prms.id, MAX_MEBS - 1);
    return 1;
  }

  _prms.ifAddr = body["meb"][id]["connect_info"]["nic_ip"];
  _prms.ebPort = std::to_string(mebPortBase + _prms.id);

  _prms.contributors = 0;
  if (body.find("drp") != body.end())
  {
    for (auto it : body["drp"].items())
    {
      unsigned drpId = it.value()["drp_id"];
      if (drpId > MAX_DRPS - 1)
      {
        fprintf(stderr, "DRP ID %d is out of range 0 - %d\n", drpId, MAX_DRPS - 1);
        return 1;
      }
      _prms.contributors |= 1ul << drpId;
    }
  }

  _prms.addrs.clear();
  _prms.ports.clear();

  if (body.find("teb") != body.end())
  {
    for (auto it : body["teb"].items())
    {
      unsigned    tebId   = it.value()["teb_id"];
      std::string address = it.value()["connect_info"]["nic_ip"];
      if (tebId > MAX_TEBS - 1)
      {
        fprintf(stderr, "TEB ID %d is out of range 0 - %d\n", tebId, MAX_TEBS - 1);
        return 1;
      }
      _prms.addrs.push_back(address);
      _prms.ports.push_back(std::string(std::to_string(mrqPortBase + tebId)));
    }
  }
  if (_prms.addrs.size() == 0)
  {
    fprintf(stderr, "Missing required TEB request address(es)\n");
    return 1;
  }

  printf("\nParameters of Monitor Event Builder ID %d:\n",  _prms.id);
  printf("  Thread core numbers:        %d, %d\n",          _prms.core[0], _prms.core[1]);
  printf("  Partition:                  %d\n",              _prms.partition);
  printf("  Buffer duration:            %014lx\n",          _prms.duration);
  printf("  Batch pool depth:           %d\n",              _prms.maxBuffers);
  printf("  Max # of entries per batch: %d\n",              _prms.maxEntries);
  printf("\n");
  printf("  MRQ port range: %d - %d\n", mrqPortBase, mrqPortBase + MAX_MEBS - 1);
  printf("  MEB port range: %d - %d\n", mebPortBase, mebPortBase + MAX_MEBS - 1);
  printf("\n");

  return 0;
}


using namespace Pds;


void usage(char* progname)
{
  printf("Usage: %s -C <collection server>"
                   "-p <partition> "
                   "[-P <partition name>] "
                   "-n <numb shm buffers> "
                   "-s <shm buffer size> "
                  "[-q <# event queues>] "
                  "[-t <tag name>] "
                  "[-d] "
                  "[-A <interface addr>] "
                  "[-Z <Run-time mon host>] "
                  "[-R <Run-time mon port>] "
                  "[-1 <core to pin App thread to>]"
                  "[-2 <core to pin other threads to>]" // Revisit: None?
                  "[-V] " // Run-time mon verbosity
                  "[-v] "
                  "[-h] "
                  "\n", progname);
}

int main(int argc, char** argv)
{
  const unsigned NO_PARTITION    = unsigned(-1u);
  const char*    tag             = 0;
  std::string    partitionTag     (PARTITION);
  std::string    collSrv          (COLL_HOST);
  const char*    rtMonHost       = RTMON_HOST;
  unsigned       rtMonPort       = RTMON_PORT_BASE;
  unsigned       rtMonPeriod     = rtMon_period;
  unsigned       rtMonVerbose    = 0;
  EbParams       prms { /* .ifAddr        = */ { }, // Network interface to use
                        /* .ebPort        = */ { },
                        /* .mrqPort       = */ { }, // Unused here
                        /* .partition     = */ NO_PARTITION,
                        /* .id            = */ -1u,
                        /* .contributors  = */ 0,   // DRPs
                        /* .addrs         = */ { }, // MonReq addr served by TEB
                        /* .ports         = */ { }, // MonReq port served by TEB
                        /* .duration      = */ epoch_duration,
                        /* .maxBuffers    = */ numberof_xferBuffers,
                        /* .maxEntries    = */ 1,   // per buffer
                        /* .numMrqs       = */ 0,   // Unused here
                        /* .maxTrSize     = */ sizeof_buffers,
                        /* .maxResultSize = */ 0,   // Unused here
                        /* .core          = */ { core_0, core_1 },
                        /* .verbose       = */ 0 };
  unsigned       sizeofEvBuffers = sizeof_buffers;
  unsigned       nevqueues       = 1;
  bool           ldist           = false;

  int c;
  while ((c = getopt(argc, argv, "p:P:n:s:q:t:dA:Z:R:C:1:2:Vvh")) != -1)
  {
    errno = 0;
    char* endPtr;
    switch (c) {
      case 'p':
        prms.partition = strtoul(optarg, &endPtr, 0);
        if (errno != 0 || endPtr == optarg) prms.partition = NO_PARTITION;
        break;
      case 'P':
        partitionTag = std::string(optarg);
        break;
      case 'n':
        sscanf(optarg, "%d", &prms.maxBuffers);
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
      case 'A':  prms.ifAddr       = optarg;                       break;
      case 'Z':  rtMonHost         = optarg;                       break;
      case 'R':  rtMonPort         = atoi(optarg);                 break;
      case 'C':  collSrv           = optarg;                       break;
      case '1':  prms.core[0]      = atoi(optarg);                 break;
      case '2':  prms.core[1]      = atoi(optarg);                 break;
      case 'v':  ++prms.verbose;                                   break;
      case 'V':  ++rtMonVerbose;                                   break;
      case 'h':                         // help
        usage(argv[0]);
        return 0;
        break;
      default:
        printf("Unrecogized parameter '%c'\n", c);
        usage(argv[0]);
        return 1;
    }
  }

  if (!prms.maxBuffers || !sizeofEvBuffers)
  {
    fprintf(stderr, "Missing parameters!\n");
    usage(argv[0]);
    return 1;
  }

  if (prms.maxBuffers < numberof_xferBuffers)
  {
    prms.maxBuffers = numberof_xferBuffers;
  }

  if (!tag)  tag = partitionTag.c_str();
  printf("Partition Tag: '%s'\n", tag);

  if (prms.partition == NO_PARTITION)
  {
    fprintf(stderr, "Partition number must be specified\n");
    return 1;
  }

  struct sigaction sigAction;

  sigAction.sa_handler = sigHandler;
  sigAction.sa_flags   = SA_RESTART;
  sigemptyset(&sigAction.sa_mask);
  if (sigaction(SIGINT, &sigAction, &lIntAction) > 0)
    printf("Couldn't set up ^C handler\n");

  // Revisit: Pinning exacerbates a race condition somewhere resulting in either
  //          'No free buffers available' or 'Dgram pool allocation failed'
  //pinThread(pthread_self(), prms.core[1]);
  StatsMonitor smon(rtMonHost,
                    rtMonPort,
                    prms.partition,
                    partitionTag,
                    rtMonPeriod,
                    rtMonVerbose);

  //pinThread(pthread_self(), prms.core[0]);
  MebApp app(collSrv, tag, sizeofEvBuffers, nevqueues, ldist, prms, smon);

  app.run();

  smon.shutdown();

  return 0;
}
