#include "psalg/shmem/XtcMonitorServer.hh"

#include "psdaq/eb/eb.hh"
#include "psdaq/eb/EbAppBase.hh"
#include "psdaq/eb/EbEvent.hh"

#include "psdaq/eb/EbLfClient.hh"

#include "psdaq/eb/utilities.hh"

#include "psdaq/service/Fifo.hh"
#include "psdaq/service/GenericPool.hh"
#include "psdaq/service/Collection.hh"
#include "psdaq/service/MetricExporter.hh"
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
static const unsigned epoch_duration       = 8;  // Revisit: 1 per xferBuffer
static const unsigned numberof_xferBuffers = 8;  // Value corresponds to ctrb:maxEvents

using namespace XtcData;
using namespace Pds::Eb;
using namespace psalg::shmem;
using namespace Pds;

using json = nlohmann::json;

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
  namespace Eb {
    struct MebParams : public EbParams
    {
      MebParams(EbParams prms, unsigned mbs, unsigned neb) :
        EbParams(prms), maxBufferSize(mbs), numEvBuffers(neb) {}

      unsigned maxBufferSize;           // Maximum built event size
      unsigned numEvBuffers;            // Number of event buffers
    };
  };

  class MyXtcMonitorServer : public XtcMonitorServer {
  public:
    MyXtcMonitorServer(const char*      tag,
                       unsigned         numberofEvQueues,
                       const MebParams& prms) :
      XtcMonitorServer(tag,
                       prms.maxBufferSize,
                       prms.numEvBuffers,
                       numberofEvQueues),
      _sizeofBuffers(prms.maxBufferSize),
      _iTeb         (0),
      _mrqTransport (prms.verbose),
      _mrqLinks     (),
      _bufFreeList  (prms.numEvBuffers),
      _id           (-1u)
    {
    }
    int connect(const MebParams& prms)
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
    Meb(const MebParams&                prms,
        std::shared_ptr<MetricExporter> exporter) :
      EbAppBase  (prms, epoch_duration, 1, prms.numEvBuffers),
      _apps      (nullptr),
      _pool      (nullptr),
      _eventCount(0),
      _prms      (prms)
    {
      std::map<std::string, std::string> labels{{"partition", std::to_string(prms.partition)}};
      exporter->add("MEB_EvtRt",  labels, MetricType::Rate,    [&](){ return _eventCount;      });
      exporter->add("MEB_EvtCt",  labels, MetricType::Counter, [&](){ return _eventCount;      });
      exporter->add("MEB_EpAlCt", labels, MetricType::Counter, [&](){ return  epochAllocCnt(); });
      exporter->add("MEB_EpFrCt", labels, MetricType::Counter, [&](){ return  epochFreeCnt();  });
      exporter->add("MEB_EvAlCt", labels, MetricType::Counter, [&](){ return  eventAllocCnt(); });
      exporter->add("MEB_EvFrCt", labels, MetricType::Counter, [&](){ return  eventFreeCnt();  });
      exporter->add("MEB_RxPdg",  labels, MetricType::Gauge,   [&](){ return  rxPending();     });
      exporter->add("MEB_BufCt",  labels, MetricType::Counter, [&](){ return  bufferCnt();     });
      exporter->add("MEB_FxUpCt", labels, MetricType::Counter, [&](){ return  fixupCnt();      });
      exporter->add("MEB_ToEvCt", labels, MetricType::Counter, [&](){ return  tmoEvtCnt();     });
    }
    virtual ~Meb()
    {
    }
  public:
    void run(MyXtcMonitorServer& apps)
    {
      pinThread(pthread_self(), _prms.core[0]);

      _apps = &apps;

      // Create pool for transferring events to MyXtcMonitorServer
      unsigned    entries = std::bitset<64>(_prms.contributors).count();
      size_t      size    = sizeof(Dgram) + entries * sizeof(Dgram*) + sizeof(unsigned);
      GenericPool pool(size, _prms.numEvBuffers);
      _pool = &pool;

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

      _apps->shutdown();

      EbAppBase::shutdown();

      _apps = nullptr;
      _pool = nullptr;
    }
    virtual void process(EbEvent* event)
    {
      if (_prms.verbose > 3)
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

      if (_prms.verbose > 2)
      {
        uint64_t pid = dg->seq.pulseId().value();
        unsigned ctl = dg->seq.pulseId().control();
        size_t   sz  = sizeof(*dg) + dg->xtc.sizeofPayload();
        unsigned src = dg->xtc.src.value();
        unsigned env = dg->env;
        unsigned svc = dg->seq.service();
        printf("MEB processed              event[%5ld]    @ "
               "%16p, ctl %02x, pid %014lx, sz %4zd, src %2d, env %08x, %3s # %2d\n",
               _eventCount, dg, ctl, pid, sz, src, env,
               svc == TransitionId::L1Accept ? "buf" : "tr", idx);
      }

      if (_apps->events(dg) == XtcMonitorServer::Handled)
      {
        Pool::free((void*)dg);
      }
    }
  private:
    MyXtcMonitorServer* _apps;
    GenericPool*        _pool;
    uint64_t            _eventCount;
    const MebParams&    _prms;
  };
};


class MebApp : public CollectionApp
{
public:
  MebApp(const std::string&               collSrv,
         const char*                      tag,
         unsigned                         numEvQueues,
         bool                             distribute,
         MebParams&                       prms,
         std::shared_ptr<MetricExporter>& exporter);
public:                                 // For CollectionApp
  json         connectionInfo() override;
  void         handleConnect(const json& msg) override;
  void         handleDisconnect(const json& msg) override;
  void         handlePhase1(const json& msg) override;
  void         handleReset(const json& msg) override;
private:
  std::string _handleConnect(const json& msg);
  int         _parseConnectionParams(const json& msg);
private:
  const char*                         _tag;
  unsigned                            _numEvQueues;
  bool                                _distribute;
  std::unique_ptr<Meb>                _meb;
  std::unique_ptr<MyXtcMonitorServer> _apps;
  std::thread                         _appThread;
  MebParams&                          _prms;
  std::shared_ptr<MetricExporter>&    _exporter;
};

MebApp::MebApp(const std::string&               collSrv,
               const char*                      tag,
               unsigned                         numEvQueues,
               bool                             distribute,
               MebParams&                       prms,
               std::shared_ptr<MetricExporter>& exporter) :
  CollectionApp(collSrv, prms.partition, "meb", prms.alias),
  _tag         (tag),
  _numEvQueues (numEvQueues),
  _distribute  (distribute),
  _prms        (prms),
  _exporter    (exporter)
{
}

json MebApp::connectionInfo()
{
  // Allow the default NIC choice to be overridden
  std::string ip = _prms.ifAddr.empty() ? getNicIp() : _prms.ifAddr;
  json body = {{"connect_info", {{"nic_ip", ip}}}};
  json bufInfo = {{"buf_count", _prms.numEvBuffers}};
  body["connect_info"].update(bufInfo);
  return body;
}

std::string MebApp::_handleConnect(const json &msg)
{
  int rc = _parseConnectionParams(msg["body"]);
  if (rc)  return std::string("Error parsing parameters");

  _meb = std::make_unique<Meb>(_prms, _exporter);
  rc = _meb->connect(_prms);
  if (rc)  return std::string("Failed MEB connect()");

  _apps = std::make_unique<MyXtcMonitorServer>(_tag, _numEvQueues, _prms);
  rc = _apps->connect(_prms);
  if (rc)  return std::string("Failed XtcMonitorServer connect()");

  _apps->distribute(_distribute);

  lRunning = 1;

  _appThread = std::thread(&Meb::run, std::ref(*_meb), std::ref(*_apps));

  return std::string{};
}

void MebApp::handleConnect(const json &msg)
{
  std::string errMsg = _handleConnect(msg);

  // Reply to collection with connect status
  json body = json({});
  if (!errMsg.empty())  body["error_info"] = errMsg;
  reply(createMsg("connect", msg["header"]["msg_id"], getId(), body));
}

void MebApp::handlePhase1(const json& msg)
{
  // Reply to collection with transition status
  json body = json({});
  std::string key = msg["header"]["key"];
  reply(createMsg(key, msg["header"]["msg_id"], getId(), body));
}

void MebApp::handleDisconnect(const json &msg)
{
  lRunning = 0;

  if (_appThread.joinable())  _appThread.join();

  // Reply to collection with connect status
  json body   = json({});
  reply(createMsg("disconnect", msg["header"]["msg_id"], getId(), body));
}

void MebApp::handleReset(const json &msg)
{
  if (_appThread.joinable())  _appThread.join();
}

static void _printGroups(unsigned groups, const u64arr_t& array)
{
  while (groups)
  {
    unsigned group = __builtin_ffs(groups) - 1;
    groups &= ~(1 << group);

    printf("%d: 0x%016lx  ", group, array[group]);
  }
  printf("\n");
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

  if (body.find("drp") == body.end())
  {
    fprintf(stderr, "Missing required DRP specs\n");
    return 1;
  }

  uint16_t groups        = 0;
  size_t   maxTrSize     = 0;
  size_t   maxBufferSize = 0;
  _prms.contributors     = 0;
  _prms.maxBufferSize    = 0;
  _prms.maxTrSize.resize(body["drp"].size());

  _prms.contractors.fill(0);
  _prms.receivers.fill(0);

  for (auto it : body["drp"].items())
  {
    unsigned drpId = it.value()["drp_id"];
    if (drpId > MAX_DRPS - 1)
    {
      fprintf(stderr, "DRP ID %d is out of range 0 - %d\n", drpId, MAX_DRPS - 1);
      return 1;
    }
    _prms.contributors |= 1ul << drpId;

    unsigned group = it.value()["det_info"]["readout"];
    if (group > NUM_READOUT_GROUPS - 1)
    {
      fprintf(stderr, "Readout group %d is out of range 0 - %d\n", group, NUM_READOUT_GROUPS - 1);
      return 1;
    }
    _prms.contractors[group] |= 1ul << drpId;
    _prms.receivers[group]    = 0;      // Unused by MEB
    groups |= 1 << group;

    _prms.maxTrSize[drpId] = size_t(it.value()["connect_info"]["max_tr_size"]);
    maxTrSize             += _prms.maxTrSize[drpId];
    maxBufferSize         += size_t(it.value()["connect_info"]["max_ev_size"]);
  }
  // shmem buffers must fit both built events and transitions of worst case size
  _prms.maxBufferSize = maxBufferSize > maxTrSize ? maxBufferSize : maxTrSize;

  if (body.find("teb") == body.end())
  {
    fprintf(stderr, "Missing required TEB specs\n");
    return 1;
  }

  _prms.addrs.clear();
  _prms.ports.clear();

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

  printf("\nParameters of MEB ID %d:\n",                     _prms.id);
  printf("  Thread core numbers:        %d, %d\n",           _prms.core[0], _prms.core[1]);
  printf("  Partition:                  %d\n",               _prms.partition);
  printf("  Bit list of contributors: 0x%016lx, cnt: %zd\n", _prms.contributors,
                                                             std::bitset<64>(_prms.contributors).count());
  printf("  Readout group contractors:  ");                  _printGroups(groups, _prms.contractors);
  printf("  Buffer duration:          0x%014lx\n",           BATCH_DURATION);
  printf("  Number of event buffers:    %d\n",               _prms.numEvBuffers);
  printf("  Max # of entries / buffer:  %d\n",               1);
  printf("  shmem buffer size:          %d\n",               _prms.maxBufferSize);
  printf("  Number of event queues:     %d\n",               _numEvQueues);
  printf("  Distribute:                 %s\n",               _distribute ? "yes" : "no");
  printf("  Tag:                        %s\n",               _tag);
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
                   "-u <alias> "
                  "[-q <# event queues>] "
                  "[-t <tag name>] "
                  "[-d] "
                  "[-A <interface addr>] "
                  "[-1 <core to pin App thread to>]"
                  "[-2 <core to pin other threads to>]" // Revisit: None?
                  "[-v] "
                  "[-h] "
                  "\n", progname);
}

int main(int argc, char** argv)
{
  const unsigned NO_PARTITION = unsigned(-1u);
  const char*    tag          = 0;
  std::string    partitionTag;
  std::string    collSrv;
  MebParams      prms { { /* .ifAddr        = */ { }, // Network interface to use
                          /* .ebPort        = */ { },
                          /* .mrqPort       = */ { }, // Unused here
                          /* .partition     = */ NO_PARTITION,
                          /* .alias         = */ { }, // Unique name passed on cmd line
                          /* .id            = */ -1u,
                          /* .contributors  = */ 0,   // DRPs
                          /* .addrs         = */ { }, // MonReq addr served by TEB
                          /* .ports         = */ { }, // MonReq port served by TEB
                          /* .maxTrSize     = */ { }, // Filled in @ connect
                          /* .maxResultSize = */ 0,   // Unused here
                          /* .numMrqs       = */ 0,   // Unused here
                          /* .core          = */ { core_0, core_1 },
                          /* .verbose       = */ 0,
                          /* .contractors   = */ 0,
                          /* .receivers     = */ 0 },
                        /* .maxBufferSize = */ 0,     // Filled in @ connect
                        /* .numEvBuffers  = */ numberof_xferBuffers };
  unsigned       nevqueues = 1;
  bool           ldist     = false;

  int c;
  while ((c = getopt(argc, argv, "p:P:n:t:q:dA:C:1:2:u:vh")) != -1)
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
        sscanf(optarg, "%d", &prms.numEvBuffers);
        break;
      case 't':
        tag = optarg;
        break;
      case 'q':
        nevqueues = strtoul(optarg, NULL, 0);
        break;
      case 'd':
        ldist = true;
        break;
      case 'A':  prms.ifAddr       = optarg;                       break;
      case 'C':  collSrv           = optarg;                       break;
      case '1':  prms.core[0]      = atoi(optarg);                 break;
      case '2':  prms.core[1]      = atoi(optarg);                 break;
      case 'u':  prms.alias        = optarg;                       break;
      case 'v':  ++prms.verbose;                                   break;
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
  if (!prms.numEvBuffers)
  {
    fprintf(stderr, "Missing '%s' parameter\n", "-n <max buffers>");
    return 1;
  }
  if (collSrv.empty())
  {
    fprintf(stderr, "Missing '%s' parameter\n", "-C <Collection server>");
    return 1;
  }
  if (prms.alias.empty()) {
    fprintf(stderr, "Missing '%s' parameter\n", "-u <Alias>");
    return 1;
  }

  if (prms.numEvBuffers < numberof_xferBuffers)
    prms.numEvBuffers = numberof_xferBuffers;

  if (!tag)  tag = partitionTag.c_str();
  printf("Partition Tag: '%s'\n", tag);

  struct sigaction sigAction;

  sigAction.sa_handler = sigHandler;
  sigAction.sa_flags   = SA_RESTART;
  sigemptyset(&sigAction.sa_mask);
  if (sigaction(SIGINT, &sigAction, &lIntAction) > 0)
    fprintf(stderr, "Failed to set up ^C handler\n");

  prometheus::Exposer exposer{"0.0.0.0:9200", "/metrics", 1};
  auto exporter = std::make_shared<MetricExporter>();

  MebApp app(collSrv, tag, nevqueues, ldist, prms, exporter);

  exposer.RegisterCollectable(exporter);

  try
  {
    app.run();
  }
  catch (std::exception& e)
  {
    fprintf(stderr, "%s\n", e.what());
  }

  app.handleReset(json({}));

  return 0;
}
