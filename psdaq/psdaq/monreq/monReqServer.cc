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
static const unsigned numberof_xferBuffers = 8;  // Revisit: Value; corresponds to ctrb:maxEvents
static const unsigned sizeof_buffers       = 64 * 1024; // Revisit

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
                       unsigned        numberofEvBuffers,
                       unsigned        numberofEvQueues,
                       const EbParams& prms) :
      XtcMonitorServer(tag,
                       sizeofBuffers,
                       numberofEvBuffers,
                       numberofEvQueues),
      _sizeofBuffers(sizeofBuffers),
      _iTeb         (0),
      _mrqTransport (prms.verbose),
      _mrqLinks     (),
      _bufFreeList  (numberofEvBuffers),
      _id           (-1u)
    {
    }
    int connect(const EbParams& prms)
    {
      _iTeb = 0;
      _id   = prms.id;
      _mrqLinks.resize(prms.addrs.size());

      unsigned numBuffers = _bufFreeList.size();
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
                  __PRETTY_FUNCTION__, addr, port);
          return rc;
        }
        if ( (rc = link->preparePoster(_id)) )
        {
          fprintf(stderr, "%s:\n  Failed to prepare Monitor link to %s:%s\n",
                  __PRETTY_FUNCTION__, addr, port);
          return rc;
        }
        _mrqLinks[link->id()] = link;

        printf("Outbound link with TEB ID %d connected\n", link->id());
      }

      for (unsigned i = 0; i < numBuffers; ++i)
      {
        if (_bufFreeList.push(i))
          fprintf(stderr, "%s:\n  _bufFreeList.push(%d) failed\n", __PRETTY_FUNCTION__, i);
        //printf("%s:\n  _bufFreeList.push(%d), count = %zd\n",
        //       __PRETTY_FUNCTION__, i, _bufFreeList.count());
      }

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

      _bufFreeList.clear();
      _id = -1;
    }

  private:
    virtual void _copyDatagram(Dgram* dg, char* buf)
    {
      //printf("_copyDatagram:   dg = %p, pid = %014lx to %p\n",
      //       dg, dg->seq.pulseId().value(), buf);

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
          // Revisit: Maybe replace this with an empty Dgram with damage set?
          abort();            // The memcpy would blow by the buffer size limit
        }

        memcpy(buf, &idg->xtc, idg->xtc.extent);
      }
      while (++ctrb != last);
    }

    virtual void _deleteDatagram(Dgram* dg)
    {
      //printf("_deleteDatagram @ %p: pid = %014lx\n",
      //       dg, dg->seq.pulseId().value());

      unsigned idx = *(unsigned*)dg->xtc.next();
      if (_bufFreeList.push(idx))
        printf("_bufFreeList.push(%d) failed, count = %zd\n", idx, _bufFreeList.count());
      //printf("_deleteDatagram: dg = %p, pid = %014lx, _bufFreeList.push(%d), count = %zd\n",
      //       dg, dg->seq.pulseId().value(), idx, _bufFreeList.count());

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

      unsigned data = _bufFreeList.front();
      _bufFreeList.pop();
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
    unsigned               _iTeb;
    EbLfClient             _mrqTransport;
    std::vector<EbLfLink*> _mrqLinks;
    FifoMT<unsigned>       _bufFreeList;
    unsigned               _id;
  };

  class Meb : public EbAppBase
  {
  public:
    Meb(const char*     tag,
        unsigned        sizeofEvBuffers,
        unsigned        numberofEvBuffers,
        unsigned        nevqueues,
        bool            dist,
        const EbParams& prms,
        StatsMonitor&   smon) :
      EbAppBase  (prms, epoch_duration, 1, numberof_xferBuffers),
      _apps      (tag, sizeofEvBuffers, numberofEvBuffers, nevqueues, prms),
      _pool      (nullptr),
      _eventCount(0),
      _verbose   (prms.verbose),
      _numBuffers(numberofEvBuffers),
      _prms      (prms),
      _dist      (dist)
    {
      smon.metric("MEB_EvtRt",  _eventCount,      StatsMonitor::RATE);
      smon.metric("MEB_EvtCt",  _eventCount,      StatsMonitor::SCALAR);
      smon.metric("MEB_EpAlCt",  epochAllocCnt(), StatsMonitor::SCALAR);
      smon.metric("MEB_EpFrCt",  epochFreeCnt(),  StatsMonitor::SCALAR);
      smon.metric("MEB_EvAlCt",  eventAllocCnt(), StatsMonitor::SCALAR);
      smon.metric("MEB_EvFrCt",  eventFreeCnt(),  StatsMonitor::SCALAR);
      smon.metric("MEB_RxPdg",   rxPending(),     StatsMonitor::SCALAR);
    }
    virtual ~Meb()
    {
    }
  public:
    int connect(const EbParams& prms)
    {
      unsigned entries = std::bitset<64>(prms.contributors).count();
      _pool = new GenericPool(sizeof(Dgram) + entries * sizeof(Dgram*) +
                              sizeof(unsigned), _numBuffers);

      int rc;
      if ( (rc = EbAppBase::connect(prms)) )
        return rc;

      if ( (rc = _apps.connect(prms)) )
        return rc;

      _apps.distribute(_dist);

      return 0;
    }
    void run()
    {
      pinThread(pthread_self(), _prms.core[0]);

      _eventCount = 0;

      while (true)
      {
        int rc;
        if (!lRunning)
        {
          if (checkEQ() == -FI_ENOTCONN)  break;
        }

        if ( (rc = EbAppBase::process()) < 0)
        {
          if (checkEQ() == -FI_ENOTCONN)  break;
        }
      }

      _apps.shutdown();

      EbAppBase::shutdown();

      delete _pool;
      _pool = nullptr;
    }
    virtual void process(EbEvent* event)
    {
      if (_verbose > 3)
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

      if (_verbose > 2)
      {
        uint64_t pid = dg->seq.pulseId().value();
        unsigned ctl = dg->seq.pulseId().control();
        size_t   sz  = sizeof(*dg) + dg->xtc.sizeofPayload();
        unsigned src = dg->xtc.src.value();
        unsigned env = dg->env;
        unsigned svc = dg->seq.service();
        printf("MEB processed              event[%4ld]    @ "
               "%16p, ctl %02x, pid %014lx, sz %4zd, src %2d, env %08x, %3s # %2d\n",
               _eventCount, dg, ctl, pid, sz, src, env,
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
    const unsigned     _numBuffers;
    const EbParams&    _prms;
    const bool         _dist;
  };
};


class MebApp : public CollectionApp
{
public:
  MebApp(const std::string& collSrv,
         const char*        tag,
         unsigned           sizeofEvBuffers,
         unsigned           maxBuffes,
         unsigned           nevqueues,
         bool               dist,
         EbParams&          prms,
         StatsMonitor&      smon);
public:                                 // For CollectionApp
  std::string nicIp() override;
  void handleConnect(const json& msg) override;
  void handleDisconnect(const json& msg); // override;
  void handleReset(const json& msg) override;
private:
  int  _handleConnect(const json& msg);
  int  _parseConnectionParams(const json& msg);
  void _shutdown();
private:
  EbParams&     _prms;
  Meb           _meb;
  StatsMonitor& _smon;
  std::thread   _appThread;
};

MebApp::MebApp(const std::string& collSrv,
               const char*        tag,
               unsigned           sizeofEvBuffers,
               unsigned           numberofEvBuffers,
               unsigned           nevqueues,
               bool               dist,
               EbParams&          prms,
               StatsMonitor&      smon) :
  CollectionApp(collSrv, prms.partition, "meb", prms.alias),
  _prms(prms),
  _meb (tag, sizeofEvBuffers, numberofEvBuffers, nevqueues, dist, prms, smon),
  _smon(smon)
{
  printf("  Tag:                        %s\n", tag);
  printf("  Max event buffer size:      %d\n", sizeofEvBuffers);
  printf("  Number of Event buffers:    %d\n", numberofEvBuffers);
  printf("  Number of Event queues:     %d\n", nevqueues);
  printf("  Distribute:                 %s\n", dist ? "yes" : "no");
}

std::string MebApp::nicIp()
{
  // Allow the default NIC choice to be overridden
  std::string ip = _prms.ifAddr.empty() ? getNicIp() : _prms.ifAddr;
  return ip;
}

int MebApp::_handleConnect(const json &msg)
{
  int rc = _parseConnectionParams(msg["body"]);
  if (rc)  return rc;

  rc = _meb.connect(_prms);
  if (rc)  return rc;

  _smon.enable();

  lRunning = 1;

  _appThread = std::thread(&Meb::run, std::ref(_meb));

  return 0;
}

void MebApp::handleConnect(const json &msg)
{
  int rc = _handleConnect(msg);

  // Reply to collection with connect status
  json body   = json({});
  json answer = (rc == 0)
              ? createMsg("connect", msg["header"]["msg_id"], getId(), body)
              : createMsg("error",   msg["header"]["msg_id"], getId(), body);
  reply(answer);
}

void MebApp::_shutdown()
{
  lRunning = 0;

  _appThread.join();

  _smon.disable();
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

  uint16_t groups = 0;
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

      groups |= 1 << unsigned(it.value()["readout"]);
    }
  }

  _prms.contractors.fill(0);
  _prms.receivers.fill(0);

  while (groups)
  {
    unsigned group = __builtin_ffs(groups) - 1;
    groups &= ~(1 << group);

    uint64_t contractors = _prms.contributors; // Revisit: Value to come from CfgDb

    if (!contractors)
    {
      fprintf(stderr, "No contributors found for readout group %d\n",
              group);
      return 1;
    }

    _prms.contractors[group] = contractors;
    _prms.receivers[group]   = 0;       // Unused by MEB
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

  printf("\nParameters of MEB ID %d:\n",                     _prms.id);
  printf("  Thread core numbers:        %d, %d\n",           _prms.core[0], _prms.core[1]);
  printf("  Partition:                  %d\n",               _prms.partition);
  printf("  Bit list of contributors: 0x%016lx, cnt: %zd\n", _prms.contributors,
                                                             std::bitset<64>(_prms.contributors).count());
  printf("  Buffer duration:          0x%014lx\n",           BATCH_DURATION);
  printf("  Buffer pool depth:          %d\n",               numberof_xferBuffers);
  printf("  Max # of entries / buffer:  %d\n",               1);
  printf("  Max transition size:        %zd\n",              _prms.maxTrSize);
  printf("\n");
  printf("  MRQ port range: %d - %d\n", mrqPortBase, mrqPortBase + MAX_MEBS - 1);
  printf("  MEB port range: %d - %d\n", mebPortBase, mebPortBase + MAX_MEBS - 1);
  printf("\n");

  return 0;
}


using namespace Pds;


void usage(char* progname)
{
  printf("Usage: %s -C <collection server> "
                   "-p <partition> "
                   "-P <partition name> "
                   "-n <numb shm buffers> "
                   "-s <shm buffer size> "
                   "-u <alias> "
                  "[-q <# event queues>] "
                  "[-t <tag name>] "
                  "[-d] "
                  "[-A <interface addr>] "
                  "-Z <Run-time mon host> "
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
  std::string    partitionTag;
  std::string    collSrv;
  const char*    rtMonHost;
  unsigned       rtMonPort       = RTMON_PORT_BASE;
  unsigned       rtMonPeriod     = rtMon_period;
  unsigned       rtMonVerbose    = 0;
  EbParams       prms { /* .ifAddr        = */ { }, // Network interface to use
                        /* .ebPort        = */ { },
                        /* .mrqPort       = */ { }, // Unused here
                        /* .partition     = */ NO_PARTITION,
                        /* .alias         = */ { }, // Unique name passed on cmd line
                        /* .id            = */ -1u,
                        /* .contributors  = */ 0,   // DRPs
                        /* .addrs         = */ { }, // MonReq addr served by TEB
                        /* .ports         = */ { }, // MonReq port served by TEB
                        /* .maxTrSize     = */ sizeof_buffers,
                        /* .maxResultSize = */ 0,   // Unused here
                        /* .numMrqs       = */ 0,   // Unused here
                        /* .core          = */ { core_0, core_1 },
                        /* .verbose       = */ 0,
                        /* .contractors   = */ 0,
                        /* .receivers     = */ 0 };
  unsigned       sizeofEvBuffers = sizeof_buffers;
  unsigned       numberOfBuffers = numberof_xferBuffers;
  unsigned       nevqueues       = 1;
  bool           ldist           = false;

  int c;
  while ((c = getopt(argc, argv, "p:P:n:s:q:t:dA:Z:R:C:1:2:u:Vvh")) != -1)
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
        sscanf(optarg, "%d", &numberOfBuffers);
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
      case 'u':  prms.alias        = optarg;                       break;
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

  if (!numberOfBuffers)
  {
    fprintf(stderr, "Missing '%s' parameter\n", "-n <max buffers>");
    return 1;
  }
  if (!sizeofEvBuffers)
  {
    fprintf(stderr, "Missing '%s' parameter\n", "-s <size of event buffers>");
    return 1;
  }

  if (prms.partition == NO_PARTITION)
  {
    fprintf(stderr, "Missing '%s' parameter\n", "-p <Partition number>");
    return 1;
  }
  if (partitionTag.empty())
  {
    fprintf(stderr, "Missing '%s' parameter\n", "-P <Partition name>");
    return 1;
  }
  if (collSrv.empty())
  {
    fprintf(stderr, "Missing '%s' parameter\n", "-C <Collection server>");
    return 1;
  }
  if (!rtMonHost)
  {
    fprintf(stderr, "Missing '%s' parameter\n", "-Z <Run-Time Monitoring host>");
    return 1;
  }
  if (prms.alias.empty()) {
    fprintf(stderr, "Missing '%s' parameter\n", "-u <Alias>");
    return 1;
  }

  if (numberOfBuffers < numberof_xferBuffers)
    numberOfBuffers = numberof_xferBuffers;

  if (!tag)  tag = partitionTag.c_str();
  printf("Partition Tag: '%s'\n", tag);

  struct sigaction sigAction;

  sigAction.sa_handler = sigHandler;
  sigAction.sa_flags   = SA_RESTART;
  sigemptyset(&sigAction.sa_mask);
  if (sigaction(SIGINT, &sigAction, &lIntAction) > 0)
    printf("Couldn't set up ^C handler\n");

  pinThread(pthread_self(), prms.core[1]);
  StatsMonitor smon(rtMonHost,
                    rtMonPort,
                    prms.partition,
                    rtMonPeriod,
                    rtMonVerbose);
  smon.startup();

  MebApp app(collSrv, tag, sizeofEvBuffers, numberOfBuffers, nevqueues, ldist, prms, smon);

  try
  {
    app.run();
  }
  catch (std::exception& e)
  {
    fprintf(stderr, "%s\n", e.what());
  }

  smon.shutdown();

  return 0;
}
