#include "EbAppBase.hh"

#include "BatchManager.hh"
#include "EbEvent.hh"

#include "EbLfClient.hh"
#include "EbLfServer.hh"

#include "utilities.hh"
#include "StatsMonitor.hh"

#include "psdaq/service/Collection.hh"
#include "xtcdata/xtc/Dgram.hh"

#include <stdio.h>
#include <unistd.h>                     // For getopt()
#include <cstring>
#include <climits>
#include <csignal>
#include <bitset>
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

    class Teb : public EbAppBase
    {
    public:
      Teb(const EbParams& prms, StatsMonitor& statsMon);
    public:
      int      connect(const EbParams&);
      void     run();
    public:                         // For EventBuilder
      virtual
      void     process(EbEvent* event);
    public:
      void     post(const Batch*, uint64_t& receivers);
    private:
      std::vector<EbLfLink*> _l3Links;
      EbLfServer             _mrqTransport;
      std::vector<EbLfLink*> _mrqLinks;
      BatchManager           _batMan;
      unsigned               _id;
      const unsigned         _verbose;
    private:
      uint64_t               _receivers;
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
  EbAppBase    (prms, BATCH_DURATION, MAX_ENTRIES, MAX_BATCHES),
  _l3Links     (),
  _mrqTransport(prms.verbose),
  _mrqLinks    (),
  _batMan      (prms.maxResultSize),
  _id          (-1),
  _verbose     (prms.verbose),
  _receivers   (0),
  _eventCount  (0),
  _batchCount  (0),
  _prms        (prms),
  _l3Transport (prms.verbose)
{
  smon.metric("TEB_EvtRt",  _eventCount,             StatsMonitor::RATE);
  smon.metric("TEB_EvtCt",  _eventCount,             StatsMonitor::SCALAR);
  smon.metric("TEB_BatCt",  _batchCount,             StatsMonitor::SCALAR);
  smon.metric("TEB_BtAlCt", _batMan.batchAllocCnt(), StatsMonitor::SCALAR);
  smon.metric("TEB_BtFrCt", _batMan.batchFreeCnt(),  StatsMonitor::SCALAR);
  smon.metric("TEB_BtWtg",  _batMan.batchWaiting(),  StatsMonitor::SCALAR);
  smon.metric("TEB_EpAlCt",  epochAllocCnt(),        StatsMonitor::SCALAR);
  smon.metric("TEB_EpFrCt",  epochFreeCnt(),         StatsMonitor::SCALAR);
  smon.metric("TEB_EvAlCt",  eventAllocCnt(),        StatsMonitor::SCALAR);
  smon.metric("TEB_EvFrCt",  eventFreeCnt(),         StatsMonitor::SCALAR);
  smon.metric("TEB_TxPdg",  _l3Transport.pending(),  StatsMonitor::SCALAR);
  smon.metric("TEB_RxPdg",   rxPending(),            StatsMonitor::SCALAR);
}

int Teb::connect(const EbParams& prms)
{
  int rc;
  if ( (rc = EbAppBase::connect(prms)) )
    return rc;

  _id = prms.id;
  _l3Links.resize(prms.addrs.size());

  void*  region  = _batMan.batchRegion();
  size_t regSize = _batMan.batchRegionSize();

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
  pinThread(pthread_self(), _prms.core[0]);

  _receivers  = 0;
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

  _batMan.dump();
  _batMan.shutdown();

  _id = -1;
}

// Ultimately, this method is provided by the users and is loaded from a
// sharable library loaded during a configure transition.
static
uint16_t decide(const Dgram* ctrb, uint32_t* result, size_t sizeofPayload)
{
  if (result)
  {
    const uint32_t* input = reinterpret_cast<uint32_t*>(ctrb->xtc.payload());

    result[WRT_IDX] |= input[0] == 0xdeadbeef ? 1 : 0;
    result[MON_IDX] |= input[1] == 0x12345678 ? 1 : 0;
  }
  return 0;
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

  const Dgram* dg = event->creator();
  if (ImmData::rsp(ImmData::flg(event->parameter())) == ImmData::Response)
  {
    uint64_t pid    = dg->seq.pulseId().value();
    Batch*   batch  = _batMan.fetch();
    if (!batch || batch->expired(pid))
    {
      if (batch)  post(batch, _receivers);

      batch = _batMan.allocate(pid);
      assert(batch);                    // Null "can't" happen
    }

    Dgram*    rdg    = new(batch->allocate()) ResultDgram(*dg, _id);
    uint32_t* result = reinterpret_cast<uint32_t*>(rdg->xtc.payload());

    // Present event contributions to "user" code for building a result datagram
    const EbContribution** const  last = event->end();
    const EbContribution*  const* ctrb = event->begin();
    do
    {
      uint16_t damage = decide(*ctrb, result, rdg->xtc.sizeofPayload());
      if (damage)  rdg->xtc.damage.increase(damage);
    }
    while (++ctrb != last);

    // Accumulate the list of ctrbs to this batch
    _receivers |= event->receivers();   // Cleared after posting, below

    if (rdg->seq.isEvent())
    {
      if (result[MON_IDX])
      {
        uint64_t data;
        int      rc = _mrqTransport.poll(&data);
        result[MON_IDX] = (rc < 0) ? 0 : data;
        if ((rc > 0) && (rc = _mrqLinks[ImmData::src(data)]->postCompRecv()) )
        {
          fprintf(stderr, "%s:\n  Failed to post CQ buffers: %d\n",
                  __PRETTY_FUNCTION__, rc);
        }
      }
    }
    else                                // Non-event
    {
      result[WRT_IDX] = 1;              // Always write out transitions
      result[MON_IDX] = 1;              // Always monitor transitions

      post(batch, _receivers);
    }

    if (_verbose > 2) // || result[1])
    {
      unsigned idx = batch->index();
      uint64_t pid = rdg->seq.pulseId().value();
      unsigned ctl = rdg->seq.pulseId().control();
      size_t   sz  = sizeof(*rdg) + rdg->xtc.sizeofPayload();
      unsigned src = rdg->xtc.src.value();
      unsigned env = rdg->env;
      printf("TEB processed              result  [%4d] @ "
             "%16p, ctl %02x, pid %014lx, sz %4zd, src %2d, env %08x, res [%08x, %08x]\n",
             idx, rdg, ctl, pid, sz, src, env, result[0], result[1]);
    }
  }
  else
  {
    // Present event contributions to "user" code for handling
    const EbContribution** const  last = event->end();
    const EbContribution*  const* ctrb = event->begin();
    do
    {
      decide(*ctrb, nullptr, 0);
    }
    while (++ctrb != last);

    // Even though no response is being emitted for this event, in the case that
    // it's a transition, any in-progress batch must be flushed
    if (!dg->seq.isEvent())
    {
      Batch* batch = _batMan.fetch();
      if (batch)  post(batch, _receivers);
    }

    if (_verbose > 2)
    {
      uint64_t pid = dg->seq.pulseId().value();
      unsigned ctl = dg->seq.pulseId().control();
      size_t   sz  = sizeof(*dg) + dg->xtc.sizeofPayload();
      unsigned src = dg->xtc.src.value();
      unsigned env = dg->env;
      printf("TEB processed           non-event         @ "
             "%16p, ctl %02x, pid %014lx, sz %4zd, src %02d, env %08x\n",
             dg, ctl, pid, sz, src, env);
    }
  }
}

void Teb::post(const Batch* batch, uint64_t& receivers)
{
  _batMan.flush();

  uint32_t    idx    = batch->index();
  uint64_t    data   = ImmData::value(ImmData::Buffer, _id, idx);
  size_t      extent = batch->extent();
  unsigned    offset = idx * _batMan.maxBatchSize();
  const void* buffer = batch->buffer();
  uint64_t    destns = receivers;

  while (destns)
  {
    unsigned  dst  = __builtin_ffsl(destns) - 1;
    EbLfLink* link = _l3Links[dst];

    destns &= ~(1 << dst);

    if (_verbose)
    {
      uint64_t pid    = batch->id();
      void*    rmtAdx = (void*)link->rmtAdx(offset);
      printf("TEB posts           %6ld result  [%4d] @ "
             "%16p,         pid %014lx,               sz %4zd, dst %2d @ %16p\n",
             _batchCount, idx, buffer, pid, extent, dst, rmtAdx);
    }

    if (link->post(buffer, extent, offset, data) < 0)  break;
  }

  receivers = 0;                        // Clear to start a new accumulation

  ++_batchCount;

  // Revisit: The following deallocation constitutes a race with the posts to
  // the transport above as the batch's memory cannot be allowed to be reused
  // for a subsequent batch before the transmit completes.  Waiting for
  // completion here would impact performance.  Since there are many batches,
  // and only one is active at a time, a previous batch will have completed
  // transmitting before the next one starts (or the subsequent transmit won't
  // start), thus making it safe to "pre-delete" it here.
  _batMan.release(batch);
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
  CollectionApp(collSrv, prms.partition, "teb", prms.alias),
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
  _prms.id = body["teb"][id]["teb_id"];
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

  _prms.groups = 1 << _prms.partition;  // Revisit: Value to come from CfgDb
  unsigned groups = _prms.groups;
  if (groups == 0)
  {
    fprintf(stderr, "No readout groups are enabled\n");
    return 1;
  }
  while (groups)
  {
    unsigned group = __builtin_ffs(groups) - 1;
    groups &= ~(1 << group);

    uint64_t contractors = _prms.contributors; // Revisit: Value to come from CfgDb
    uint64_t receivers   = _prms.contributors; // Revisit: Value to come from CfgDb

    //if (group == 0)  contractors &= ~(1ul << 1); // Take out DRP 1 for a test

    if (!contractors)
    {
      fprintf(stderr, "No trigger %s found for readout group %d\n",
              "input data contractors", group);
      return 1;
    }
    if (!receivers)
    {
      fprintf(stderr, "No trigger %s found for readout group %d\n",
              "result receivers", group);
      return 1;
    }
    if ((contractors & receivers) != contractors)
    {
      fprintf(stderr, "Readout group %d's receivers contract (%016lx) "
                      "must also contain its contractors (%016lx)\n",
              group, receivers, contractors);
      return 1;
    }

    _prms.contractors[group] = contractors;
    _prms.receivers[group]   = receivers;
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
  printf("  Enabled readout groups:   0x%02x\n",             _prms.groups);
  printf("  Bit list of contributors: 0x%016lx, cnt: %zd\n", _prms.contributors,
                                                             std::bitset<64>(_prms.contributors).count());
  printf("  Number of MEB requestors:   %d\n",               _prms.numMrqs);
  printf("  Batch duration:           0x%014lx = %ld uS\n",  BATCH_DURATION, BATCH_DURATION);
  printf("  Batch pool depth:           %d\n",               MAX_BATCHES);
  printf("  Max # of entries / batch:   %d\n",               MAX_ENTRIES);
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
  fprintf(stderr, " %-22s %s (required)\n",           "-u <alias>",
          "Alias for teb process");
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
  const char*    rtMonHost    = nullptr;
  unsigned       rtMonPort    = RTMON_PORT_BASE;
  unsigned       rtMonPeriod  = rtMon_period;
  unsigned       rtMonVerbose = 0;
  EbParams       prms { /* .ifAddr        = */ { }, // Network interface to use
                        /* .ebPort        = */ { },
                        /* .mrqPort       = */ { },
                        /* .partition     = */ NO_PARTITION,
                        /* .alias         = */ { }, // Unique name passed on cmd line
                        /* .id            = */ -1u,
                        /* .contributors  = */ 0,   // DRPs
                        /* .addrs         = */ { }, // Result dst addr served by Ctrbs
                        /* .ports         = */ { }, // Result dst port served by Ctrbs
                        /* .maxTrSize     = */ max_contrib_size,
                        /* .maxResultSize = */ max_result_size,
                        /* .numMrqs       = */ 0,   // Number of Mon requestors
                        /* .core          = */ { core_0, core_1 },
                        /* .verbose       = */ 0,
                        /* .contractors   = */ 0,
                        /* .receivers     = */ 0,
                        /* .groups        = */ 0 };

  while ((op = getopt(argc, argv, "C:p:A:Z:R:1:2:u:h?vV")) != -1)
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
      case 'u':  prms.alias      = optarg;             break;
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
  if (prms.alias.empty()) {
    fprintf(stderr, "Missing '%s' parameter\n", "-u <Alias>");
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
  smon.startup();

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
