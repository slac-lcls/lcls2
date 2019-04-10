#include "psdaq/eb/EbAppBase.hh"

#include "psdaq/eb/BatchManager.hh"
#include "psdaq/eb/EbEvent.hh"

#include "psdaq/eb/EbLfClient.hh"
#include "psdaq/eb/EbLfServer.hh"

#include "psdaq/eb/utilities.hh"
#include "psdaq/eb/StatsMonitor.hh"

#include "psdaq/service/Collection.hh"
#include "xtcdata/xtc/Dgram.hh"

#include <stdio.h>
#include <unistd.h>                     // For getopt()
#include <cstring>
#include <climits>
#include <csignal>
#include <bitset>
#include <chrono>
#include <atomic>
#include <vector>
#include <cassert>
#include <iostream>
#include <exception>


using namespace XtcData;
using namespace Pds;

static const int      core_0           = 10; // devXXX: 10, devXX:  7, accXX:  9
static const int      core_1           = 11; // devXXX: 11, devXX: 19, accXX: 21
static const unsigned rtMon_period     = 1; // Seconds
static const size_t   header_size      = sizeof(Dgram);
static const size_t   input_extent     = 2; // Revisit: Number of "L3" input  data words
static const size_t   result_extent    = 2; // Revisit: Number of "L3" result data words
static const size_t   max_contrib_size = header_size + input_extent  * sizeof(uint32_t);
static const size_t   max_result_size  = header_size + result_extent * sizeof(uint32_t);

using TimePoint_t = std::chrono::steady_clock::time_point;
using Duration_t  = std::chrono::steady_clock::duration;
using us_t        = std::chrono::microseconds;
using ns_t        = std::chrono::nanoseconds;

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

    class ResultDgram : public Dgram
    {
    public:
      ResultDgram(const Transition& transition_, unsigned id) :
        Dgram(transition_, Xtc(TypeId(TypeId::Data, 0), Src(id, Level::Event)))
      {
        xtc.alloc(sizeof(_data));

        for (unsigned i = 0; i < result_extent; ++i)
          _data[i] = 0;
      }
    private:
      uint32_t _data[result_extent];
    };

    class Teb;

    class TebBatchManager : public BatchManager
    {
    public:
      TebBatchManager(const EbParams& prms, Teb& app) :
        BatchManager(prms.duration, prms.maxBuffers, prms.maxEntries, prms.maxResultSize),
        _app(app)
      {
      }
    public:                             // For BatchManager
      virtual void post(const Batch* batch);
    private:
      Teb& _app;
    };

    class Teb : public EbAppBase
    {
    public:
      Teb(const EbParams& prms, StatsMonitor& statsMon);
    public:
      int      connect(const EbParams&);
      void     run();
    public:
      void     post(const Batch*);
      void     post(const Dgram*);
    public:                         // For EventBuilder
      virtual
      void     process(EbEvent* event);
    public:                         // Ultimately loaded from a shareable
      uint16_t handle(const Dgram* ctrb, uint32_t* result, size_t sizeofPayload);
    private:
      std::vector<EbLfLink*> _l3Links;
      EbLfServer             _mrqTransport;
      std::vector<EbLfLink*> _mrqLinks;
      TebBatchManager        _batchManager;
      unsigned               _id;
      const unsigned         _verbose;
    private:
      uint64_t               _dstList;
    private:
      uint64_t               _eventCount;
      uint64_t               _batchCount;
    private:
      const EbParams&        _prms;
      EbLfClient             _l3Transport;
    };
  };
};


using namespace Pds::Eb;

Teb::Teb(const EbParams& prms, StatsMonitor& smon) :
  EbAppBase    (prms),
  _l3Links     (),
  _mrqTransport(prms.verbose),
  _mrqLinks    (),
  _batchManager(prms, *this),
  _id          (-1),
  _verbose     (prms.verbose),
  _dstList     (0),
  _eventCount  (0),
  _batchCount  (0),
  _prms        (prms),
  _l3Transport (prms.verbose)
{
  smon.registerIt("TEB_EvtRt",  _eventCount,                   StatsMonitor::RATE);
  smon.registerIt("TEB_EvtCt",  _eventCount,                   StatsMonitor::SCALAR);
  smon.registerIt("TEB_BatCt",  _batchCount,                   StatsMonitor::SCALAR);
  smon.registerIt("TEB_BtAlCt", _batchManager.batchAllocCnt(), StatsMonitor::SCALAR);
  smon.registerIt("TEB_BtFrCt", _batchManager.batchFreeCnt(),  StatsMonitor::SCALAR);
  smon.registerIt("TEB_BtWtg",  _batchManager.batchWaiting(),  StatsMonitor::SCALAR);
  smon.registerIt("TEB_EpAlCt",  epochAllocCnt(),              StatsMonitor::SCALAR);
  smon.registerIt("TEB_EpFrCt",  epochFreeCnt(),               StatsMonitor::SCALAR);
  smon.registerIt("TEB_EvAlCt",  eventAllocCnt(),              StatsMonitor::SCALAR);
  smon.registerIt("TEB_EvFrCt",  eventFreeCnt(),               StatsMonitor::SCALAR);
  smon.registerIt("TEB_TxPdg",  _l3Transport.pending(),        StatsMonitor::SCALAR);
  smon.registerIt("TEB_RxPdg",   rxPending(),                  StatsMonitor::SCALAR);
}

int Teb::connect(const EbParams& prms)
{
  int rc;
  if ( (rc = EbAppBase::connect(prms)) )
    return rc;

  _id = prms.id;
  _l3Links.resize(prms.addrs.size());

  void*  region  = _batchManager.batchRegion();
  size_t regSize = _batchManager.batchRegionSize();

  for (unsigned i = 0; i < prms.addrs.size(); ++i)
  {
    const char*    addr = prms.addrs[i].c_str();
    const char*    port = prms.ports[i].c_str();
    EbLfLink*      link;
    const unsigned tmo(120000);         // Milliseconds
    if ( (rc = _l3Transport.connect(addr, port, tmo, &link)) )
    {
      fprintf(stderr, "%s:\n  Error connecting to Ctrb at %s:%s\n",
              __PRETTY_FUNCTION__, addr, port);
      return rc;
    }
    if ( (rc = link->preparePoster(_id, region, regSize)) )
    {
      fprintf(stderr, "%s:\n  Failed to prepare link with Ctrb at %s:%s\n",
              __PRETTY_FUNCTION__, addr, port);
      return rc;
    }
    _l3Links[link->id()] = link;

    printf("Outbound link with Ctrb ID %d connected\n", link->id());
  }

  if ( (rc = _mrqTransport.initialize(prms.ifAddr, prms.mrqPort, prms.numMrqs)) )
  {
    fprintf(stderr, "%s:\n  Failed to initialize MonReq EbLfServer\n",
            __PRETTY_FUNCTION__);
    return rc;
  }

  _mrqLinks.resize(prms.numMrqs);

  for (unsigned i = 0; i < prms.numMrqs; ++i)
  {
    EbLfLink*      link;
    const unsigned tmo(120000);         // Milliseconds
    if ( (rc = _mrqTransport.connect(&link, tmo)) )
    {
      fprintf(stderr, "%s:\n  Error connecting to MonReq %d\n",
              __PRETTY_FUNCTION__, i);
      return rc;
    }

    if ( (rc = link->preparePender(prms.id)) )
    {
      fprintf(stderr, "%s:\n  Failed to prepare MonReq %d\n",
              __PRETTY_FUNCTION__, i);
      return rc;
    }
    _mrqLinks[link->id()] = link;
    if ( (rc = link->postCompRecv()) )
    {
      fprintf(stderr, "%s:\n  Failed to post CQ buffers: %d\n",
              __PRETTY_FUNCTION__, rc);
    }

    printf("Inbound link with MonReq ID %d connected\n", link->id());
  }

  return 0;
}

void Teb::run()
{
  //pinThread(task()->parameters().taskID(), _prms.core[1]);
  //pinThread(pthread_self(),                _prms.core[1]);

  //start();                              // Start the event timeout timer

  pinThread(pthread_self(),                _prms.core[0]);

  _dstList    = 0;
  _eventCount = 0;
  _batchCount = 0;

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

  //cancel();                             // Stop the event timeout timer

  for (auto it = _mrqLinks.begin(); it != _mrqLinks.end(); ++it)
  {
    _mrqTransport.shutdown(*it);
  }
  _mrqLinks.clear();
  _mrqTransport.shutdown();

  for (auto it = _l3Links.begin(); it != _l3Links.end(); ++it)
  {
    _l3Transport.shutdown(*it);
  }
  _l3Links.clear();

  EbAppBase::shutdown();

  _batchManager.dump();
  _batchManager.shutdown();

  _id = -1;
}

void Teb::process(EbEvent* event)
{
  // Accumulate output datagrams (result) from the event builder into a batch
  // datagram.  When the batch completes, post it to the contributors.

  if (_verbose > 3)
  {
    static unsigned cnt = 0;
    printf("Teb::process event dump:\n");
    event->dump(++cnt);
  }
  ++_eventCount;

  if (ImmData::rsp(ImmData::flg(event->parameter())) == ImmData::Response)
  {
    const Dgram* dg     = event->creator();
    Batch*       batch  = _batchManager.locate(dg->seq.pulseId().value());  assert(batch); // This may post a batch
    Dgram*       rdg    = new(batch->buffer()) ResultDgram(*dg, _id);
    uint32_t*    result = (uint32_t*)rdg->xtc.payload();

    _dstList |= event->contract();      // Accumulate the list of ctrbs to this batch
                                        // Cleared after posting, below

    // Present event contributions to "user" code for building a result datagram
    const EbContribution** const  last = event->end();
    const EbContribution*  const* ctrb = event->begin();
    do
    {
      uint16_t damage = handle(*ctrb, result, rdg->xtc.sizeofPayload());
      if (damage)  rdg->xtc.damage.increase(damage);
    }
    while (++ctrb != last);

    if (rdg->seq.isEvent())
    {
      if (result[1])
      {
        uint64_t data;
        int      rc = _mrqTransport.poll(&data);
        result[1] = (rc < 0) ? 0 : data;
        if (rc > 0)  _mrqLinks[ImmData::src(data)]->postCompRecv();
      }
    }
    else                                // Non-event
    {
      _batchManager.post(batch);
      _batchManager.flush();
    }

    if ((_verbose > 2)) // || result[1])
    {
      unsigned idx = batch->index();
      uint64_t pid = rdg->seq.pulseId().value();
      unsigned ctl = rdg->seq.pulseId().control();
      size_t   sz  = sizeof(*rdg) + rdg->xtc.sizeofPayload();
      unsigned src = rdg->xtc.src.value();
      printf("TEB processed              result  [%4d] @ "
             "%16p, ctl %02x, pid %014lx, sz %4zd, src %2d, res [%08x, %08x]\n",
             idx, rdg, ctl, pid, sz, src, result[0], result[1]);
    }
  }
  else
  {
    // Present event contributions to "user" code for handling
    const EbContribution** const  last = event->end();
    const EbContribution*  const* ctrb = event->begin();
    do
    {
      handle(*ctrb, nullptr, 0);
    }
    while (++ctrb != last);

    if (_verbose > 2)
    {
      const Dgram* dg = event->creator();
      uint64_t pid = dg->seq.pulseId().value();
      unsigned ctl = dg->seq.pulseId().control();
      size_t   sz  = sizeof(*dg) + dg->xtc.sizeofPayload();
      unsigned src = dg->xtc.src.value();
      printf("TEB processed           non-event         @ "
             "%16p, ctl %02x, pid %014lx, sz %4zd, src %02d\n",
             dg, ctl, pid, sz, src);
    }
  }
}

// Ultimately, this method is provided by the users and is loaded from a
// sharable library loaded during a configure transition.
uint16_t Teb::handle(const Dgram* ctrb, uint32_t* result, size_t sizeofPayload)
{
  if (result)
  {
    const uint32_t* input = (uint32_t*)ctrb->xtc.payload();
    result[0] |= input[0] == 0xdeadbeef ? 1 : 0;
    result[1] |= input[1] == 0x12345678 ? 1 : 0;
  }
  return 0;
}

void TebBatchManager::post(const Batch* batch)
{
  _app.post(batch);
}

void Teb::post(const Batch* batch)
{
  uint32_t    idx    = batch->index();
  uint64_t    data   = ImmData::value(ImmData::Buffer, _id, idx);
  size_t      extent = batch->extent();
  unsigned    offset = idx * _batchManager.maxBatchSize();
  const void* buffer = batch->batch();
  uint64_t    mask   = _dstList;

  auto t0(std::chrono::steady_clock::now());
  while (mask)
  {
    unsigned  dst  = __builtin_ffsl(mask) - 1;
    EbLfLink* link = _l3Links[dst];

    mask &= ~(1 << dst);

    if (_verbose)
    {
      uint64_t pid    = batch->id();
      void*    rmtAdx = (void*)link->rmtAdx(offset);
      printf("TEB posts           %6ld result  [%4d] @ "
             "%16p,         pid %014lx, sz %4zd, dst %2d @ %16p\n",
             _batchCount, idx, buffer, pid, extent, link->id(), rmtAdx);
    }

    if (link->post(buffer, extent, offset, data) < 0)  break;
  }
  auto t1(std::chrono::steady_clock::now());

  _dstList = 0;                         // Clear to start a new accumulation

  ++_batchCount;

  // Revisit: The following deallocation constitutes a race with the posts to
  // the transport above as the batch's memory cannot be allowed to be reused
  // for a subsequent batch before the transmit completes.  Waiting for
  // completion here would impact performance.  Since there are many batches,
  // and only one is active at a time, a previous batch will have completed
  // transmitting before the next one starts (or the subsequent transmit won't
  // start), thus making it safe to "pre-delete" it here.
  _batchManager.release(batch);
}

void Teb::post(const Dgram* nonEvent)
{
  assert(false);                        // Unused virtual method
}


class TebApp : public CollectionApp
{
public:
  TebApp(const std::string& collSrv, EbParams&, StatsMonitor&);
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
  Teb           _teb;
  StatsMonitor& _smon;
  std::thread   _appThread;
};

TebApp::TebApp(const std::string& collSrv,
               EbParams&          prms,
               StatsMonitor&      smon) :
  CollectionApp(collSrv, prms.partition, "teb"),
  _prms(prms),
  _teb(prms, smon),
  _smon(smon)
{
}

std::string TebApp::nicIp()
{
  // Allow the default NIC choice to be overridden
  std::cout<<"custom nicIp\n";
  std::string ip = _prms.ifAddr.empty() ? getNicIp() : _prms.ifAddr;
  return ip;
}

int TebApp::_handleConnect(const json &msg)
{
  int rc = _parseConnectionParams(msg["body"]);
  if (rc)  return rc;

  rc = _teb.connect(_prms);
  if (rc)  return rc;

  _smon.enable();

  lRunning = 1;

  _appThread = std::thread(&Teb::run, std::ref(_teb));

  return 0;
}

void TebApp::handleConnect(const json &msg)
{
  int rc = _handleConnect(msg);

  // Reply to collection with connect status
  json body   = json({});
  json answer = (rc == 0)
              ? createMsg("connect", msg["header"]["msg_id"], getId(), body)
              : createMsg("error",   msg["header"]["msg_id"], getId(), body);
  reply(answer);
}

void TebApp::_shutdown()
{
  lRunning = 0;

  _appThread.join();

  _smon.disable();
}

void TebApp::handleDisconnect(const json &msg)
{
  _shutdown();

  // Reply to collection with connect status
  json body   = json({});
  json answer = createMsg("disconnect", msg["header"]["msg_id"], getId(), body);
  reply(answer);
}

void TebApp::handleReset(const json &msg)
{
  _shutdown();
}

int TebApp::_parseConnectionParams(const json& body)
{
  const unsigned numPorts    = MAX_DRPS + MAX_TEBS + MAX_TEBS + MAX_MEBS;
  const unsigned tebPortBase = TEB_PORT_BASE + numPorts * _prms.partition;
  const unsigned drpPortBase = DRP_PORT_BASE + numPorts * _prms.partition;
  const unsigned mrqPortBase = MRQ_PORT_BASE + numPorts * _prms.partition;

  std::string id = std::to_string(getId());
  _prms.id       = body["teb"][id]["teb_id"];
  if (_prms.id >= MAX_TEBS)
  {
    fprintf(stderr, "TEB ID %d is out of range 0 - %d\n", _prms.id, MAX_TEBS - 1);
    return 1;
  }

  _prms.ifAddr  = body["teb"][id]["connect_info"]["nic_ip"];
  _prms.ebPort  = std::to_string(tebPortBase + _prms.id);
  _prms.mrqPort = std::to_string(mrqPortBase + _prms.id);

  _prms.contributors = 0;
  _prms.addrs.clear();
  _prms.ports.clear();

  if (body.find("drp") != body.end())
  {
    for (auto it : body["drp"].items())
    {
      unsigned    drpId   = it.value()["drp_id"];
      std::string address = it.value()["connect_info"]["nic_ip"];
      if (drpId > MAX_DRPS - 1)
      {
        fprintf(stderr, "DRP ID %d is out of range 0 - %d\n", drpId, MAX_DRPS - 1);
        return 1;
      }
      _prms.contributors |= 1ul << drpId;
      _prms.addrs.push_back(address);
      _prms.ports.push_back(std::string(std::to_string(drpPortBase + drpId)));
    }
  }
  if (_prms.addrs.size() == 0)
  {
    fprintf(stderr, "Missing required DRP address(es)\n");
    return 1;
  }

  _prms.numMrqs = 0;
  if (body.find("meb") != body.end())
  {
    for (auto it : body["meb"].items())
    {
      _prms.numMrqs++;
    }
  }

  printf("\nParameters of TEB ID %d:\n",                     _prms.id);
  printf("  Thread core numbers:        %d, %d\n",           _prms.core[0], _prms.core[1]);
  printf("  Partition:                  %d\n",               _prms.partition);
  printf("  Bit list of contributors:   %016lx, cnt: %zd\n", _prms.contributors,
                                                             std::bitset<64>(_prms.contributors).count());
  printf("  Number of Monitor EBs:      %d\n",               _prms.numMrqs);
  printf("  Batch duration:             %014lx = %ld uS\n",  _prms.duration, _prms.duration);
  printf("  Batch pool depth:           %d\n",               _prms.maxBuffers);
  printf("  Max # of entries / batch:   %d\n",               _prms.maxEntries);
  printf("  Max result     Dgram size:  %zd\n",              _prms.maxResultSize);
  printf("  Max transition Dgram size:  %zd\n",              _prms.maxTrSize);
  printf("\n");
  printf("  TEB port range: %d - %d\n", tebPortBase, tebPortBase + MAX_TEBS - 1);
  printf("  DRP port range: %d - %d\n", drpPortBase, drpPortBase + MAX_DRPS - 1);
  printf("  MRQ port range: %d - %d\n", mrqPortBase, mrqPortBase + MAX_MEBS - 1);
  printf("\n");

  return 0;
}


static void usage(char *name, char *desc, const EbParams& prms)
{
  fprintf(stderr, "Usage:\n");
  fprintf(stderr, "  %s [OPTIONS]\n", name);

  if (desc)
    fprintf(stderr, "\n%s\n", desc);

  fprintf(stderr, "\nOptions:\n");

  fprintf(stderr, " %-22s %s (default: %s)\n",        "-A <interface_addr>",
          "IP address of the interface to use",       "libfabric's 'best' choice");

  fprintf(stderr, " %-22s %s (required)\n",           "-C <address>",
          "Collection server");
  fprintf(stderr, " %-22s %s (required)\n",           "-p <partition number>",
          "Partition number");
  fprintf(stderr, " %-22s %s (required)\n",           "-Z <address>",
          "Run-time monitoring ZMQ server host");
  fprintf(stderr, " %-22s %s (default: %d)\n",        "-R <port>",
          "Run-time monitoring ZMQ server port",      RTMON_PORT_BASE);
  fprintf(stderr, " %-22s %s (default: %d)\n",        "-m <seconds>",
          "Run-time monitoring printout period",      rtMon_period);
  fprintf(stderr, " %-22s %s (default: %d)\n",        "-1 <core>",
          "Core number for pinning App thread to",    core_0);
  fprintf(stderr, " %-22s %s (default: %d)\n",        "-2 <core>",
          "Core number for pinning other threads to", core_1);

  fprintf(stderr, " %-22s %s\n", "-v", "enable debugging output (repeat for increased detail)");
  fprintf(stderr, " %-22s %s\n", "-h", "display this help output");
}


int main(int argc, char **argv)
{
  const unsigned NO_PARTITION = unsigned(-1u);
  int            op           = 0;
  std::string    collSrv;
  const char*    rtMonHost;
  unsigned       rtMonPort    = RTMON_PORT_BASE;
  unsigned       rtMonPeriod  = rtMon_period;
  unsigned       rtMonVerbose = 0;
  EbParams       prms { /* .ifAddr        = */ { }, // Network interface to use
                        /* .ebPort        = */ { },
                        /* .mrqPort       = */ { },
                        /* .partition     = */ NO_PARTITION,
                        /* .id            = */ -1u,
                        /* .contributors  = */ 0,   // DRPs
                        /* .addrs         = */ { }, // Result dst addr served by Ctrbs
                        /* .ports         = */ { }, // Result dst port served by Ctrbs
                        /* .duration      = */ BATCH_DURATION,
                        /* .maxBuffers    = */ MAX_BATCHES,
                        /* .maxEntries    = */ MAX_ENTRIES,
                        /* .numMrqs       = */ 0,   // Number of Mon requestors
                        /* .maxTrSize     = */ max_contrib_size,
                        /* .maxResultSize = */ max_result_size,
                        /* .core          = */ { core_0, core_1 },
                        /* .verbose       = */ 0 };

  while ((op = getopt(argc, argv, "C:p:A:Z:R:1:2:h?vV")) != -1)
  {
    switch (op)
    {
      case 'C':  collSrv         = optarg;             break;
      case 'p':  prms.partition  = std::stoi(optarg);  break;
      case 'A':  prms.ifAddr     = optarg;             break;
      case 'Z':  rtMonHost       = optarg;             break;
      case 'R':  rtMonPort       = atoi(optarg);       break;
      case '1':  prms.core[0]    = atoi(optarg);       break;
      case '2':  prms.core[1]    = atoi(optarg);       break;
      case 'v':  ++prms.verbose;                       break;
      case 'V':  ++rtMonVerbose;                       break;
      case '?':
      case 'h':
      default:
        usage(argv[0], (char*)"Trigger Event Builder application", prms);
        return 1;
    }
  }

  if (prms.partition == NO_PARTITION)
  {
    fprintf(stderr, "Missing '%s' parameter\n", "-p <Partition number>");
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

  struct sigaction sigAction;

  sigAction.sa_handler = sigHandler;
  sigAction.sa_flags   = SA_RESTART;
  sigemptyset(&sigAction.sa_mask);
  if (sigaction(SIGINT, &sigAction, &lIntAction) > 0)
    printf("Couldn't set up ^C handler\n");

  // Event builder sorts contributions into a time ordered list
  // Then calls the user's process() with complete events to build the result datagram
  // Post completed result batches to the outlet

  // Iterate over contributions in the batch
  // Event build them according to their trigger group

  pinThread(pthread_self(), prms.core[1]);
  StatsMonitor smon(rtMonHost,
                    rtMonPort,
                    prms.partition,
                    rtMonPeriod,
                    rtMonVerbose);

  pinThread(pthread_self(), prms.core[0]);
  TebApp app(collSrv, prms, smon);

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
