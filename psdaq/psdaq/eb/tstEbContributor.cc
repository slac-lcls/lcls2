#include "psdaq/eb/MebContributor.hh"
#include "psdaq/eb/TebContributor.hh"
#include "psdaq/eb/EbCtrbInBase.hh"

#include "psdaq/eb/utilities.hh"
#include "psdaq/eb/StatsMonitor.hh"
#include "psdaq/eb/EbLfClient.hh"
#include "psdaq/eb/utilities.hh"

#include "psdaq/service/GenericPoolW.hh"
#include "psdaq/service/Collection.hh"
#include "xtcdata/xtc/Dgram.hh"

#include <stdio.h>
#include <unistd.h>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <cstdint>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <iostream>                     // std::cin, std::cout
#include <bitset>
#include <exception>

//#define SINGLE_EVENTS                   // Define to take one event at a time by hitting <CR>

using namespace XtcData;
using namespace Pds;

static const unsigned CLS              = 64;   // Cache Line Size
static const int      core_0           = 10;   // devXXX: 10, devXX:  7, accXX:  9
static const int      core_1           = 11;   // devXXX: 11, devXX: 19, accXX: 21
static const unsigned rtMon_period     = 1;    // Seconds
static const size_t   header_size      = sizeof(Dgram);
static const size_t   input_extent     = 2;    // Revisit: Number of "L3" input  data words
static const size_t   result_extent    = 2;    // Revisit: Number of "L3" result data words
static const size_t   max_contrib_size = header_size + input_extent  * sizeof(uint32_t);
static const size_t   max_result_size  = header_size + result_extent * sizeof(uint32_t);
static const unsigned mon_buf_cnt      = 8;    // Revisit: Corresponds to monReqServer::numberofEvBuffers
static const size_t   mon_buf_size     = 64 * 1024; // Revisit: Corresponds to monReqServer:sizeofBuffers option
static const size_t   mon_trSize       = 64 * 1024; // Revisit: Corresponds to monReqServer:??? option

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

    class Input : public Dgram
    {
    public:
      Input() : Dgram() {}
      Input(const Sequence& seq_, const Xtc& xtc_) :
        Dgram()
      {
        seq = seq_;
        xtc = xtc_;
      }
    public:
      PoolDeclare;
    public:
      uint64_t id() const { return seq.pulseId().value(); }
    };

    class DrpSim
    {
    public:
      DrpSim(unsigned maxBatches,
             unsigned maxEntries,
             size_t   maxEvtSize);
    public:
      void            startup(unsigned id, void** base, size_t* size);
      void            shutdown();
    public:
      const Dgram*    genInput();
    public:
      const uint64_t& allocPending() const { return _allocPending; }
    private:
      const unsigned _maxBatches;
      const unsigned _maxEntries;
      const size_t   _maxEvtSz;
      const Xtc      _xtc;
      uint64_t       _pid;
      GenericPoolW*  _pool;
    private:
      enum { TrUnknown,
             TrReset,
             TrConfigure,       TrUnconfigure,
             TrEnable,          TrDisable,
             TrL1Accept };
      int            _trId;
    private:
      uint64_t       _allocPending;
      uint64_t       _startCnt;
    };

    class EbCtrbIn : public EbCtrbInBase
    {
    public:
      EbCtrbIn(const TebCtrbParams& prms,
               StatsMonitor&        smon);
      virtual ~EbCtrbIn() {}
    public:
      int      connect(const TebCtrbParams&, MebContributor*);
      void     shutdown();
    public:                             // For EbCtrbInBase
      virtual void process(const Dgram* result, const void* input);
    private:
      MebContributor* _mebCtrb;
    private:
      uint64_t        _eventCount;
    };

    class EbCtrbApp : public TebContributor
    {
    public:
      EbCtrbApp(const TebCtrbParams& prms,
                StatsMonitor&        smon);
      virtual ~EbCtrbApp() {}
    public:
      DrpSim& drpSim() { return _drpSim; }
    public:
      void     run(EbCtrbIn&);
    private:
      DrpSim               _drpSim;
      const TebCtrbParams& _prms;
    private:
      uint64_t             _eventCount;
      uint64_t             _inFlightCnt;
    };
  };
};


using namespace Pds::Eb;

DrpSim::DrpSim(unsigned maxBatches,
               unsigned maxEntries,
               size_t   maxEvtSize) :
  _maxBatches  (maxBatches),
  _maxEntries  (maxEntries),
  _maxEvtSz    (maxEvtSize),
  _xtc         (),
  _pid         (0),
  _pool        (nullptr),
  _trId        (TrUnknown),
  _allocPending(0)
{
}

void DrpSim::startup(unsigned id, void** base, size_t* size)
{
  _pid          = 0x01000000000003ul; // Something non-zero and not on a batch boundary
  _trId         = TrUnknown;
  _allocPending = 0;

  const_cast<Xtc&>(_xtc) = Xtc(TypeId(TypeId::Data, 0), Src(id));

  _pool = new GenericPoolW(sizeof(Entry) + _maxEvtSz, _maxBatches * _maxEntries, CLS);
  assert(_pool);

  *base = _pool->buffer();
  *size = _pool->size();
}

void DrpSim::shutdown()
{
  if (_pool)
  {
    _pool->stop();

    printf("\nDrpSim Input data pool:\n");
    _pool->dump();

    delete _pool;
    _pool = nullptr;
  }
}

const Dgram* DrpSim::genInput()
{
  static const TransitionId::Value trId[] =
    { TransitionId::Unknown,
      TransitionId::Reset,
      TransitionId::Configure,       TransitionId::Unconfigure,
      TransitionId::Enable,          TransitionId::Disable,
      TransitionId::L1Accept };
  const uint32_t bounceRate = 16 * 1024 * 1024 - 1;
  const uint64_t bounceRem  = 0x01000000000003ul % bounceRate;

  if       (_trId == TrDisable)               _trId  = TrEnable;
  else if  (_trId != TrL1Accept)              _trId += 2;
  else if ((_pid % bounceRate) == bounceRem)  _trId  = TrDisable;

  // Fake datagram header
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  const Sequence seq(Sequence::Event, trId[_trId], TimeStamp(ts), PulseId(_pid));

  ++_allocPending;
  void* buffer = _pool->alloc(sizeof(Input));
  --_allocPending;
  if (!buffer)  return (Dgram*)buffer;
  Input* idg = ::new(buffer) Input(seq, _xtc);

  size_t inputSize = input_extent * sizeof(uint32_t);

  // Revisit: Here is where L3 trigger input information would be inserted into the datagram,
  //          whatever that will look like
  uint32_t* payload = (uint32_t*)idg->xtc.alloc(inputSize);
  payload[0] = _pid %   3 == 0 ? 0xdeadbeef : 0xabadcafe;
  payload[1] = _pid % 119 == 0 ? 0x12345678 : 0;

#ifdef SINGLE_EVENTS
  payload[1] = 0x12345678;

  _pid += _maxEntries;
#else
  _pid += 1; //27; // Revisit: fi_tx_attr.iov_limit = 6 = 1 + max # events per batch
  //_pid = ts.tv_nsec / 1000;     // Revisit: Not guaranteed to be the same across systems
#endif

  return idg;
}


EbCtrbIn::EbCtrbIn(const TebCtrbParams& prms,
                   StatsMonitor&        smon) :
  EbCtrbInBase(prms),
  _mebCtrb    (nullptr),
  _eventCount (0)
{
  smon.registerIt("CtbI_EvtCt", _eventCount,  StatsMonitor::SCALAR);
  smon.registerIt("CtbI_RxPdg",  rxPending(), StatsMonitor::SCALAR);
}

int EbCtrbIn::connect(const TebCtrbParams& tebPrms, MebContributor* mebCtrb)
{
  _eventCount = 0;

  _mebCtrb = mebCtrb;

  return EbCtrbInBase::connect(tebPrms);
}

void EbCtrbIn::shutdown()
{
  if (_mebCtrb)  _mebCtrb->shutdown();
}

void EbCtrbIn::process(const Dgram* result, const void* appPrm)
{
  const Input* input = (const Input*)appPrm;
  uint64_t     pid   = result->seq.pulseId().value();

  if (pid != input->seq.pulseId().value())
  {
    fprintf(stderr, "Result, Input pulse ID mismatch - got: %014lx, expected: %014lx\n",
           pid, input->seq.pulseId().value());
    abort();
  }

  if (_mebCtrb)
  {
    if (result->seq.isEvent())            // L1Accept
    {
      uint32_t* response = (uint32_t*)result->xtc.payload();

      if (response[1])  _mebCtrb->post(input, response[1]);
    }
    else                                  // Other Transition
    {
      //printf("Non-event rcvd: pid = %014lx, service = %d\n", pid, result->seq.service());

      _mebCtrb->post(input);
    }
  }

  // Revisit: Race condition
  // There's a race when the input datagram is posted to the monitoring node(s).
  // It is nominally safe to delete the DG only when the post has completed.
  delete input;

  ++_eventCount;
}


EbCtrbApp::EbCtrbApp(const TebCtrbParams& prms,
                     StatsMonitor&        smon) :
  TebContributor(prms),
  _drpSim       (prms.maxBatches, prms.maxEntries, prms.maxInputSize),
  _prms         (prms),
  _eventCount   (0),
  _inFlightCnt  (0)
{
  smon.registerIt("CtbO_EvtRt",  _eventCount,            StatsMonitor::RATE);
  smon.registerIt("CtbO_EvtCt",  _eventCount,            StatsMonitor::SCALAR);
  smon.registerIt("CtbO_BatCt",   batchCount(),          StatsMonitor::SCALAR);
  smon.registerIt("CtbO_BtAlCt",  batchAllocCnt(),       StatsMonitor::SCALAR);
  smon.registerIt("CtbO_BtFrCt",  batchFreeCnt(),        StatsMonitor::SCALAR);
  smon.registerIt("CtbO_BtWtg",   batchWaiting(),        StatsMonitor::SCALAR);
  smon.registerIt("CtbO_AlPdg",  _drpSim.allocPending(), StatsMonitor::SCALAR);
  smon.registerIt("CtbO_TxPdg",   txPending(),           StatsMonitor::SCALAR);
  smon.registerIt("CtbO_InFlt",  _inFlightCnt,           StatsMonitor::SCALAR);
}

void EbCtrbApp::run(EbCtrbIn& in)
{
  _eventCount  = 0;
  _inFlightCnt = 0;

  TebContributor::startup(in);

  pinThread(pthread_self(), _prms.core[0]);

#ifdef SINGLE_EVENTS
  printf("Hit <return> for an event\n");
#endif

  while (lRunning)
  {
#ifdef SINGLE_EVENTS
    std::string blank;
    std::getline(std::cin, blank);
#endif
    //usleep(100000);

    const Dgram* input = _drpSim.genInput();
    if (!input)  continue;

    if (TebContributor::process(input, (const void*)input)) // 2nd arg is returned with the result
      ++_eventCount;

    _inFlightCnt = inFlightCnt();
  }

  TebContributor::shutdown();

  _drpSim.shutdown();

  in.shutdown();        // Revisit: The 'in' base class is shut down in TebCtrb
}


class CtrbApp : public CollectionApp
{
public:
  CtrbApp(const std::string& collSrv, TebCtrbParams&, MebCtrbParams&, StatsMonitor&);
public:                                 // For CollectionApp
  void handleAlloc(const json& msg) override;
  void handleConnect(const json& msg) override;
  void handleDisconnect(const json& msg); // override;
  void handleReset(const json& msg) override;
private:
  int  _handleConnect(const json& msg);
  int  _parseConnectionParams(const json& msg);
  void _shutdown();
private:
  TebCtrbParams& _tebPrms;
  MebCtrbParams& _mebPrms;
  EbCtrbApp      _tebCtrb;
  MebContributor _mebCtrb;
  EbCtrbIn       _inbound;
  StatsMonitor&  _smon;
  std::thread    _appThread;
};

CtrbApp::CtrbApp(const std::string& collSrv,
                 TebCtrbParams&     tebPrms,
                 MebCtrbParams&     mebPrms,
                 StatsMonitor&      smon) :
  CollectionApp(collSrv, tebPrms.partition, "drp"),
  _tebPrms(tebPrms),
  _mebPrms(mebPrms),
  _tebCtrb(tebPrms, smon),
  _mebCtrb(mebPrms),
  _inbound(tebPrms, smon),
  _smon(smon)
{
}

void CtrbApp::handleAlloc(const json& msg)
{
  // Allow the default NIC choice to be overridden
  std::string nicIp = _tebPrms.ifAddr.empty() ? getNicIp() : _tebPrms.ifAddr;
  std::cout << "nic ip  " << nicIp << '\n';
  json body = {{getLevel(), {{"connect_info", {{"nic_ip", nicIp}}}}}};
  json answer = createMsg("alloc", msg["header"]["msg_id"], getId(), body);
  reply(answer);
}

int CtrbApp::_handleConnect(const json &msg)
{
  int rc = _parseConnectionParams(msg["body"]);
  if (rc)  return rc;

  rc = _tebCtrb.connect(_tebPrms);
  if (rc)  return rc;

  void*  base;
  size_t size;
  _tebCtrb.drpSim().startup(_tebPrms.id, &base, &size);

  MebContributor* mebCtrb = nullptr;
  if (_mebPrms.addrs.size() != 0)
  {
    int rc = _mebCtrb.connect(_mebPrms, base, size);
    if (rc)  return rc;

    mebCtrb = &_mebCtrb;

    _smon.registerIt("MCtbO_EvtCt", _mebCtrb.eventCount(), StatsMonitor::SCALAR);
  }

  rc = _inbound.connect(_tebPrms, mebCtrb);
  if (rc)  return rc;

  _smon.enable();

  lRunning = 1;

  _appThread = std::thread(&EbCtrbApp::run, std::ref(_tebCtrb), std::ref(_inbound));

  return 0;
}

void CtrbApp::handleConnect(const json &msg)
{
  int rc = _handleConnect(msg);

  // Reply to collection with connect status
  json body   = json({});
  json answer = (rc == 0)
              ? createMsg("connect", msg["header"]["msg_id"], getId(), body)
              : createMsg("error",   msg["header"]["msg_id"], getId(), body);
  reply(answer);
}

void CtrbApp::_shutdown()
{
  lRunning = 0;

  _appThread.join();

  _smon.disable();
}

void CtrbApp::handleDisconnect(const json &msg)
{
  _shutdown();

  // Reply to collection with connect status
  json body   = json({});
  json answer = createMsg("disconnect", msg["header"]["msg_id"], getId(), body);
  reply(answer);
}

void CtrbApp::handleReset(const json &msg)
{
  _shutdown();
}

int CtrbApp::_parseConnectionParams(const json& body)
{
  const unsigned numPorts    = MAX_DRPS + MAX_TEBS + MAX_TEBS + MAX_MEBS;
  const unsigned tebPortBase = TEB_PORT_BASE + numPorts * _tebPrms.partition;
  const unsigned drpPortBase = DRP_PORT_BASE + numPorts * _tebPrms.partition;
  const unsigned mebPortBase = MEB_PORT_BASE + numPorts * _tebPrms.partition;

  std::string id = std::to_string(getId());
  _tebPrms.id    = body["drp"][id]["drp_id"];
  _mebPrms.id    = _tebPrms.id;
  if (_tebPrms.id >= MAX_DRPS)
  {
    fprintf(stderr, "DRP ID %d is out of range 0 - %d\n", _tebPrms.id, MAX_DRPS - 1);
    return 1;
  }

  _tebPrms.ifAddr = body["drp"][id]["connect_info"]["nic_ip"];
  _tebPrms.port   = std::to_string(drpPortBase + _tebPrms.id);

  _tebPrms.builders = 0;
  _tebPrms.addrs.clear();
  _tebPrms.ports.clear();

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
      _tebPrms.builders |= 1ul << tebId;
      _tebPrms.addrs.push_back(address);
      _tebPrms.ports.push_back(std::string(std::to_string(tebPortBase + tebId)));
    }
  }
  if (_tebPrms.addrs.size() == 0)
  {
    fprintf(stderr, "Missing required TEB address(es)\n");
    return 1;
  }

  _mebPrms.addrs.clear();
  _mebPrms.ports.clear();

  if (body.find("meb") != body.end())
  {
    for (auto it : body["meb"].items())
    {
      unsigned    mebId   = it.value()["meb_id"];
      std::string address = it.value()["connect_info"]["nic_ip"];
      if (mebId > MAX_MEBS - 1)
      {
        fprintf(stderr, "MEB ID %d is out of range 0 - %d\n", mebId, MAX_MEBS - 1);
        return 1;
      }
      _mebPrms.addrs.push_back(address);
      _mebPrms.ports.push_back(std::string(std::to_string(mebPortBase + mebId)));
    }
  }

  printf("\nParameters of Contributor ID %d:\n",             _tebPrms.id);
  printf("  Thread core numbers:        %d, %d\n",           _tebPrms.core[0], _tebPrms.core[1]);
  printf("  Partition:                  %d\n",               _tebPrms.partition);
  printf("  Bit list of TEBs:           %016lx, cnt: %zd\n", _tebPrms.builders,
                                                             std::bitset<64>(_tebPrms.builders).count());
  printf("  Number of Monitor EBs:      %zd\n",              _mebPrms.addrs.size());
  printf("  Batch duration:             %014lx = %ld uS\n",  _tebPrms.duration, _tebPrms.duration);
  printf("  Batch pool depth:           %d\n",               _tebPrms.maxBatches);
  printf("  Max # of entries / batch:   %d\n",               _tebPrms.maxEntries);
  printf("  Max TEB contribution size:  %zd\n",              _tebPrms.maxInputSize);
  printf("  Max MEB event        size:  %zd\n",              _mebPrms.maxEvSize);
  printf("  Max MEB transition   size:  %zd\n",              _mebPrms.maxTrSize);
  printf("\n");
  printf("  TEB port range: %d - %d\n", tebPortBase, tebPortBase + MAX_TEBS - 1);
  printf("  DRP port range: %d - %d\n", drpPortBase, drpPortBase + MAX_DRPS - 1);
  printf("  MEB port range: %d - %d\n", mebPortBase, mebPortBase + MAX_MEBS - 1);
  printf("\n");

  return 0;
}


static
void usage(char *name, char *desc, const TebCtrbParams& prms)
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
          "Core number for pinning App thread to",    prms.core[0]);
  fprintf(stderr, " %-22s %s (default: %d)\n",        "-2 <core>",
          "Core number for pinning other threads to", prms.core[1]);

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
  TebCtrbParams  tebPrms { /* .ifAddr        = */ { }, // Network interface to use
                           /* .port          = */ { }, // Port served to TEBs
                           /* .partition     = */ NO_PARTITION,
                           /* .id            = */ -1u,
                           /* .builders      = */ 0,   // TEBs
                           /* .addrs         = */ { },
                           /* .ports         = */ { },
                           /* .duration      = */ BATCH_DURATION,
                           /* .maxBatches    = */ MAX_BATCHES,
                           /* .maxEntries    = */ MAX_ENTRIES,
                           /* .maxInputSize  = */ max_contrib_size,
                           /* .core          = */ { core_0, core_1 },
                           /* .verbose       = */ 0 };
  MebCtrbParams  mebPrms { /* .addrs         = */ { },
                           /* .ports         = */ { },
                           /* .id            = */ tebPrms.id,
                           /* .maxEvents     = */ mon_buf_cnt,
                           /* .maxEvSize     = */ mon_buf_size,
                           /* .maxTrSize     = */ mon_trSize,
                           /* .verbose       = */ 0 };

  while ((op = getopt(argc, argv, "C:p:A:Z:R:o:1:2:h?vV")) != -1)
  {
    switch (op)
    {
      case 'C':  collSrv            = optarg;             break;
      case 'p':  tebPrms.partition  = std::stoi(optarg);  break;
      case 'A':  tebPrms.ifAddr     = optarg;             break;
      case 'Z':  rtMonHost          = optarg;             break;
      case 'R':  rtMonPort          = atoi(optarg);       break;
      case '1':  tebPrms.core[0]    = atoi(optarg);       break;
      case '2':  tebPrms.core[1]    = atoi(optarg);       break;
      case 'v':  ++tebPrms.verbose;
                 ++mebPrms.verbose;                       break;
      case 'V':  ++rtMonVerbose;                          break;
      case '?':
      case 'h':
        usage(argv[0], (char*)"Test DRP event contributor", tebPrms);
        return 1;
    }
  }

  if (tebPrms.partition == NO_PARTITION)
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

  // Revisit: Fix maxBatches to what will fit in the ImmData idx field?
  if (tebPrms.maxBatches - 1 > ImmData::MaxIdx)
  {
    fprintf(stderr, "Batch index ([0, %d]) can exceed available range ([0, %d])\n",
            tebPrms.maxBatches - 1, ImmData::MaxIdx);
    abort();
  }

  // Revisit: Fix maxEntries to equal duration?
  if (tebPrms.maxEntries > tebPrms.duration)
  {
    fprintf(stderr, "More batch entries (%u) requested than definable "
            "in the batch duration (%lu)\n",
            tebPrms.maxEntries, tebPrms.duration);
    abort();
  }

  struct sigaction sigAction;

  sigAction.sa_handler = sigHandler;
  sigAction.sa_flags   = SA_RESTART;
  sigemptyset(&sigAction.sa_mask);
  if (sigaction(SIGINT, &sigAction, &lIntAction) > 0)
    printf("Couldn't set up ^C handler\n");

  pinThread(pthread_self(), tebPrms.core[1]);
  StatsMonitor smon(rtMonHost,
                    rtMonPort,
                    tebPrms.partition,
                    rtMonPeriod,
                    rtMonVerbose);

  pinThread(pthread_self(), tebPrms.core[0]);
  CtrbApp app(collSrv, tebPrms, mebPrms, smon);

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
