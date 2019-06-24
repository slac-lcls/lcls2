#include "MebContributor.hh"
#include "TebContributor.hh"
#include "EbCtrbInBase.hh"

#include "utilities.hh"
#include "EbLfClient.hh"
#include "utilities.hh"

#include "psdaq/service/GenericPoolW.hh"
#include "psdaq/service/Collection.hh"
#include "psdaq/service/MetricExporter.hh"
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

using json = nlohmann::json;

static const unsigned CLS              = 64;   // Cache Line Size
static const int      core_0           = 10;   // devXXX: 10, devXX:  7, accXX:  9
static const int      core_1           = 11;   // devXXX: 11, devXX: 19, accXX: 21
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
      Input(const uint16_t readoutGroup, const Sequence& seq_, const Xtc& xtc_) :
        Dgram()
      {
        seq = seq_;
        env = readoutGroup;
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
      DrpSim(size_t maxEvtSize);
    public:
      void            startup(unsigned id, void** base, size_t* size, uint16_t readoutGroup);
      void            shutdown();
      void            stop();
      int             transition(TransitionId::Value transition, uint64_t pid);
    public:
      const Dgram*    generate();
      void            release(const Input*);
    public:
      const uint64_t& allocPending() const { return _allocPending; }
    private:
      const size_t                     _maxEvtSz;
      uint16_t                         _readoutGroup;
      const Xtc                        _xtc;
      uint64_t                         _pid;
      //GenericPoolW*                    _pool;
      GenericPool*                     _pool;
    private:
      TransitionId::Value              _trId;
      mutable std::mutex               _compLock;
      std::condition_variable          _compCv;
      bool                             _released;
      mutable std::mutex               _trLock;
      std::condition_variable          _trCv;
      std::atomic<TransitionId::Value> _transition;
      std::atomic<uint64_t>            _trPid;
    private:
      uint64_t                         _allocPending;
    };

    class EbCtrbIn : public EbCtrbInBase
    {
    public:
      EbCtrbIn(const TebCtrbParams&            prms,
               DrpSim&                         drpSim,
               ZmqContext&                     context,
               std::shared_ptr<MetricExporter> exporter);
      virtual ~EbCtrbIn() {}
    public:
      int      connect(const TebCtrbParams&, MebContributor*);
      void     shutdown();
    public:                             // For EbCtrbInBase
      virtual void process(const Dgram* result, const void* input);
    private:
      DrpSim&         _drpSim;
      MebContributor* _mebCtrb;
      ZmqSocket       _inprocSend;
      uint64_t        _pid;
    };

    class EbCtrbApp : public TebContributor
    {
    public:
      EbCtrbApp(const TebCtrbParams&            prms,
                std::shared_ptr<MetricExporter> exporter);
      ~EbCtrbApp() {}
    public:
      DrpSim&  drpSim() { return _drpSim; }
    public:
      void     stop();
      void     run(EbCtrbIn&);
    private:
      DrpSim               _drpSim;
      const TebCtrbParams& _prms;
    };
  };
};


using namespace Pds::Eb;

DrpSim::DrpSim(size_t maxEvtSize) :
  _maxEvtSz    (maxEvtSize),
  _readoutGroup(-1),
  _xtc         (),
  _pid         (0),
  _pool        (nullptr),
  _trId        (TransitionId::NumberOf), // Invalid transition
  _compLock    (),
  _compCv      (),
  _released    (false),
  _trLock      (),
  _trCv        (),
  _transition  (_trId),
  _allocPending(0)
{
}

void DrpSim::startup(unsigned id, void** base, size_t* size, uint16_t readoutGroup)
{
  _pid          = 0;
  _trId         = TransitionId::NumberOf; // Invalid transition
  _transition   = _trId;
  _allocPending = 0;
  _readoutGroup = readoutGroup;

  const_cast<Xtc&>(_xtc) = Xtc(TypeId(TypeId::Data, 0), Src(id));

  // Avoid going into resource wait by configuring more events in the pool than
  // are necessary to fill all the batches in the batch pool, i.e., make the
  // system run out of batches before it runs out of events
  assert(_pool == nullptr);
  //_pool = new GenericPoolW(sizeof(Entry) + _maxEvtSz, (MAX_BATCHES + 1) * MAX_ENTRIES, CLS);
  _pool = new GenericPool(sizeof(Entry) + _maxEvtSz, (MAX_BATCHES + 1) * MAX_ENTRIES, CLS);
  assert(_pool);

  *base = _pool->buffer();
  *size = _pool->size();
}

void DrpSim::stop()
{
  {
    std::lock_guard<std::mutex> lock(_compLock);
    _compCv.notify_one();

    //if (_pool)  _pool->stop();
  }

  transition(TransitionId::NumberOf, 0);
}

void DrpSim::shutdown()
{
  if (_pool)
  {
    printf("\nDrpSim Input data pool:\n");
    _pool->dump();

    delete _pool;
    _pool = nullptr;
  }
}

int DrpSim::transition(TransitionId::Value transition, uint64_t pid)
{
  std::unique_lock<std::mutex> lock(_trLock);
  _trPid      = pid & ((1ul << PulseId::NumPulseIdBits) - 1);
  _transition = transition;
  _trCv.notify_one();
  return 0;
}

const Dgram* DrpSim::generate()
{
  if (_trId != TransitionId::L1Accept)
  {
    if (_trId != TransitionId::NumberOf) // Allow only one transition in the system at a time
    {
      _allocPending += 2;
      std::unique_lock<std::mutex> lock(_compLock);
      _compCv.wait(lock, [this] { return _released || !lRunning; }); // Block until transition returns
      _allocPending -= 2;
      if (!lRunning)  return nullptr;
      _released = false;
    }
    if (_trId != TransitionId::Enable)   // Wait for the next transition to be issued
    {
      _allocPending += 3;
      std::unique_lock<std::mutex> lock(_trLock);
      _trCv.wait(lock, [this] { return (_transition != TransitionId::NumberOf) || !lRunning; });
      _allocPending -= 3;
      if (!lRunning)  return nullptr;
      _trId       = _transition;
      _transition = TransitionId::NumberOf;
      _pid        = _trPid;
    }
    else
    {
      _trId = TransitionId::L1Accept;
    }
  }
  else
  {
    if (_transition != TransitionId::NumberOf)
    {
      _trId       = _transition;
      _transition = TransitionId::NumberOf; // Invalid transition

      uint64_t pid = _trPid;
      if (_pid > pid)
      {
        fprintf(stderr, "%s:\n  Pulse Id overflow: %014lx, expected < %014lx\n",
                __PRETTY_FUNCTION__, _pid, pid);
        pid = _pid;                     // Don't let PID go into the past
      }
      _pid = pid;                       // Revisit: This can potentially step on a live event
    }
  }

  ++_allocPending;
  void* buffer = _pool->alloc(sizeof(Input));
  --_allocPending;
  if (!buffer)  return (Dgram*)buffer;

  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  const Sequence seq(Sequence::Event, _trId, TimeStamp(ts), PulseId(_pid));

  Input* idg = ::new(buffer) Input(_readoutGroup, seq, _xtc);

  size_t inputSize = input_extent * sizeof(uint32_t);

  // Here is where trigger input information is inserted into the datagram
  {
    uint32_t* payload = (uint32_t*)idg->xtc.alloc(inputSize);
    payload[WRT_IDX] = (_pid %   3) == 0 ? 0xdeadbeef : 0xabadcafe;
    payload[MON_IDX] = (_pid % 119) == 0 ? 0x12345678 : 0;
    //payload[MON_IDX] = _pid & (131072 - 1);
  }

#ifdef SINGLE_EVENTS
  payload[1] = 0x12345678;

  _pid += MAX_ENTRIES;
#else
  _pid += 1;
#endif

  //const uint64_t dropRate(3000000 * 10);
  //if (_xtc.src.value() == 1)
  //{
  //  static unsigned cnt = 0;
  //  if ((_pid % dropRate) == cnt)
  //  {
  //    if (cnt++ == MAX_ENTRIES - 1)  cnt = 0;
  //    //printf("Dropping %014lx\n", _pid);
  //    _pid += 1;
  //  }
  //}

  return idg;
}

void DrpSim::release(const Input* input)
{
  if (!input->seq.isEvent())
  {
    std::lock_guard<std::mutex> lock(_compLock);
    _released = true;
    _compCv.notify_one();
  }

  delete input;
}


EbCtrbIn::EbCtrbIn(const TebCtrbParams&            prms,
                   DrpSim&                         drpSim,
                   ZmqContext&                     context,
                   std::shared_ptr<MetricExporter> exporter) :
  EbCtrbInBase(prms, exporter),
  _drpSim     (drpSim),
  _mebCtrb    (nullptr),
  _inprocSend (&context, ZMQ_PAIR),
  _pid        (0)
{
  _inprocSend.connect("inproc://drp");
}

int EbCtrbIn::connect(const TebCtrbParams& tebPrms, MebContributor* mebCtrb)
{
  _pid = 0;

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

  assert(input);

  //if (result->xtc.damage.value())
  //{
  //  fprintf(stderr, "%s:\n  Result with pulse Id %014lx has damage %04x\n",
  //          __PRETTY_FUNCTION__, pid, result->xtc.damage.value());
  //}

  if (pid != input->seq.pulseId().value())
  {
    fprintf(stderr, "%s:\n  Result, Input pulse Id mismatch - expected: %014lx, got: %014lx\n",
            __PRETTY_FUNCTION__, input->seq.pulseId().value(), pid);
    return;
  }

  if (_pid > pid)
  {
    fprintf(stderr, "%s:\n  Pulse Id didn't increase: previous = %014lx, current = %014lx\n",
            __PRETTY_FUNCTION__, _pid, pid);
    abort();
  }
  _pid = pid;

  if (_mebCtrb)
  {
    if (result->seq.isEvent())          // L1Accept
    {
      uint32_t* response = (uint32_t*)result->xtc.payload();

      if (response[MON_IDX])  _mebCtrb->post(input, response[MON_IDX]);
    }
    else                                // Other Transition
    {
      _mebCtrb->post(input);
    }
  }

  // Pass non L1 accepts to control level
  if (!result->seq.isEvent())
  {
    printf("%s:\n  Saw '%s' transition on %014lx\n",
           __PRETTY_FUNCTION__, TransitionId::name(result->seq.service()), pid);

    // Send pulseId to inproc so it gets forwarded to Collection
    _inprocSend.send(std::to_string(pid * 100)); // Convert PID back to "timestamp"
  }

  // Revisit: Race condition
  // There's a race when the input datagram is posted to the monitoring node(s).
  // It is nominally safe to delete the DG only when the post has completed.
  _drpSim.release(input);               // Return the event to the pool
}


EbCtrbApp::EbCtrbApp(const TebCtrbParams&            prms,
                     std::shared_ptr<MetricExporter> exporter) :
  TebContributor(prms, exporter),
  _drpSim       (prms.maxInputSize),
  _prms         (prms)
{
  std::map<std::string, std::string> labels{{"partition", std::to_string(prms.partition)}};
  exporter->add("TCtbO_AlPdg", labels, MetricType::Gauge, [&](){ return _drpSim.allocPending(); });
}

void EbCtrbApp::stop()
{
  _drpSim.stop();

  TebContributor::stop();
}

void EbCtrbApp::run(EbCtrbIn& in)
{
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
    //using ns_t = std::chrono::nanoseconds;
    //auto  then = std::chrono::high_resolution_clock::now();
    //auto  now  = then;
    //do
    //{
    //  now = std::chrono::high_resolution_clock::now();
    //}
    //while (std::chrono::duration_cast<ns_t>(now - then).count() < 2);

    const Dgram* input = _drpSim.generate();
    if (!input)  continue;

    void* buffer = allocate(input, input); // 2nd arg is returned with the result
    if (buffer)
    {
      // Copy entire datagram into the batch (copy ctor doesn't copy payload)
      memcpy(buffer, input, sizeof(*input) + input->xtc.sizeofPayload());

      process(static_cast<Dgram*>(buffer));
    }
  }

  TebContributor::shutdown();

  _drpSim.shutdown();

  in.shutdown();        // Revisit: The 'in' base class is shut down in TebCtrb
}


class CtrbApp : public CollectionApp
{
public:
  CtrbApp(const std::string& collSrv, TebCtrbParams&, MebCtrbParams&, std::shared_ptr<MetricExporter>);
public:                                 // For CollectionApp
  json connectionInfo() override;
  void handleConnect(const json& msg) override;
  void handleDisconnect(const json& msg) override;
  void handlePhase1(const json &msg) override;
  void handleReset(const json& msg) override;
private:
  int  _handleConnect(const json& msg);
  int  _parseConnectionParams(const json& msg);
private:
  TebCtrbParams& _tebPrms;
  MebCtrbParams& _mebPrms;
  EbCtrbApp      _tebCtrb;
  MebContributor _mebCtrb;
  EbCtrbIn       _inbound;
  std::thread    _appThread;
  bool           _shutdown;
};

CtrbApp::CtrbApp(const std::string&              collSrv,
                 TebCtrbParams&                  tebPrms,
                 MebCtrbParams&                  mebPrms,
                 std::shared_ptr<MetricExporter> exporter) :
  CollectionApp(collSrv, tebPrms.partition, "drp", tebPrms.alias),
  _tebPrms(tebPrms),
  _mebPrms(mebPrms),
  _tebCtrb(tebPrms, exporter),
  _mebCtrb(mebPrms, exporter),
  _inbound(tebPrms, _tebCtrb.drpSim(), context(), exporter),
  _shutdown(false)
{
}

json CtrbApp::connectionInfo()
{
  // Allow the default NIC choice to be overridden
  std::string ip = _tebPrms.ifAddr.empty() ? getNicIp() : _tebPrms.ifAddr;
  json body = {{"connect_info", {{"nic_ip", ip}}}};
  return body;
}

int CtrbApp::_handleConnect(const json &msg)
{
  int rc = _parseConnectionParams(msg["body"]);
  if (rc)  return rc;

  rc = _tebCtrb.connect(_tebPrms);
  if (rc)  return rc;

  void*  base;
  size_t size;
  _tebCtrb.drpSim().startup(_tebPrms.id, &base, &size, _tebPrms.readoutGroup);

  MebContributor* mebCtrb = nullptr;
  if (_mebPrms.addrs.size() != 0)
  {
    int rc = _mebCtrb.connect(_mebPrms, base, size);
    if (rc)  return rc;

    mebCtrb = &_mebCtrb;
  }

  rc = _inbound.connect(_tebPrms, mebCtrb);
  if (rc)  return rc;

  lRunning = 1;

  _appThread = std::thread(&EbCtrbApp::run, std::ref(_tebCtrb), std::ref(_inbound));

  return 0;
}

void CtrbApp::handleConnect(const json &msg)
{
  int rc = _handleConnect(msg);

  // Reply to collection with connect status
  json body = json({});
  if (rc)  body["err_info"] = "Connect error";
  reply(createMsg("connect", msg["header"]["msg_id"], getId(), body));
}

void CtrbApp::handlePhase1(const json &msg)
{
  int         rc  = 0;
  std::string key = msg["header"]["key"];
  std::string ts  = msg["header"]["msg_id"];
  uint64_t    pid = 0x01000000000003ul; // Something non-zero and not on a batch boundary
  size_t      fnd = ts.find('-');
  if (fnd != std::string::npos)
  {
    // Use base 10 to deal with leading zeros
    // Divide by 100 to allow max rate of "10 MHz" before overflow
    pid = (std::stoul(ts.substr(0, fnd), nullptr, 10) * 1000000000 +
           std::stoul(ts.substr(fnd + 1, std::string::npos), nullptr, 10)) / 100;
  }

  // Ugly hack to give TEB time to react before passing the transition down the
  // "timing stream".  Must ensure that all processes in the system have
  // completed Phase 1 before injecting Phase 2 so that Phase 2 responses don't
  // arrive in Collection intermingled with Phase 1 responses.  Keep in mind the
  // relation of this sleep() time with the Collection Phase 1 timeouts.
  sleep(key == "configure" ? 5 : 1); // Nothing to wait on that ensures TEB is done

  if      (key == "configure")
    rc = _tebCtrb.drpSim().transition(TransitionId::Configure, pid);
  else if (key == "unconfigure")
    rc = _tebCtrb.drpSim().transition(TransitionId::Unconfigure, pid);
  else if (key == "enable")
    rc = _tebCtrb.drpSim().transition(TransitionId::Enable, pid);
  else if (key == "disable")
    rc = _tebCtrb.drpSim().transition(TransitionId::Disable, pid);

  // Reply to collection with transition status
  json body = json({});
  if (rc)  body["err_info"] = "Phase 1 error";
  reply(createMsg(key, msg["header"]["msg_id"], getId(), body));
}

void CtrbApp::handleDisconnect(const json &msg)
{
  lRunning = 0;

  _tebCtrb.stop();

  if (_appThread.joinable())  _appThread.join();

  // Reply to collection with connect status
  json body = json({});
  reply(createMsg("disconnect", msg["header"]["msg_id"], getId(), body));

  _shutdown = true;
}

void CtrbApp::handleReset(const json &msg)
{
  if (!_shutdown)
  {
    _tebCtrb.stop();

    if (_appThread.joinable())  _appThread.join();

    _shutdown = true;
  }
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

  unsigned group = body["drp"][id]["det_info"]["readout"];
  if (group > NUM_READOUT_GROUPS - 1)
  {
    fprintf(stderr, "Readout group %d is out of range 0 - %d\n", group, NUM_READOUT_GROUPS - 1);
    return 1;
  }
  _tebPrms.readoutGroup = 1 << group;
  _tebPrms.contractor   = _tebPrms.readoutGroup; // Revisit: Value to come from CfgDb

  //if (_tebPrms.id == 1)  _tebPrms.contractor = 2; // Let DRP 1 be a receiver only for group 0

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
  printf("  Readout group receipient: 0x%02x\n",             _tebPrms.readoutGroup);
  printf("  Readout group contractor: 0x%02x\n",             _tebPrms.contractor);
  printf("  Bit list of TEBs:         0x%016lx, cnt: %zd\n", _tebPrms.builders,
                                                             std::bitset<64>(_tebPrms.builders).count());
  printf("  Number of MEBs:             %zd\n",              _mebPrms.addrs.size());
  printf("  Batch duration:           0x%014lx = %ld uS\n",  BATCH_DURATION, BATCH_DURATION);
  printf("  Batch pool depth:           %d\n",               MAX_BATCHES);
  printf("  Max # of entries / batch:   %d\n",               MAX_ENTRIES);
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
  fprintf(stderr, " %-22s %s (required)\n",           "-u <alias>",
          "Alias for drp process");
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
  TebCtrbParams  tebPrms { /* .ifAddr        = */ { }, // Network interface to use
                           /* .port          = */ { }, // Port served to TEBs
                           /* .partition     = */ NO_PARTITION,
                           /* .alias         = */ { }, // Unique name from cmd line
                           /* .id            = */ -1u,
                           /* .builders      = */ 0,   // TEBs
                           /* .addrs         = */ { },
                           /* .ports         = */ { },
                           /* .maxInputSize  = */ max_contrib_size,
                           /* .core          = */ { core_0, core_1 },
                           /* .verbose       = */ 0,
                           /* .groups        = */ 0,
                           /* .contractor    = */ 0 };
  MebCtrbParams  mebPrms { /* .addrs         = */ { },
                           /* .ports         = */ { },
                           /* .partition     = */ NO_PARTITION,
                           /* .id            = */ tebPrms.id,
                           /* .maxEvents     = */ mon_buf_cnt,
                           /* .maxEvSize     = */ mon_buf_size,
                           /* .maxTrSize     = */ mon_trSize,
                           /* .verbose       = */ 0 };

  while ((op = getopt(argc, argv, "C:p:A:1:2:u:h?v")) != -1)
  {
    switch (op)
    {
      case 'C':  collSrv            = optarg;             break;
      case 'p':  tebPrms.partition  = std::stoi(optarg);  break;
      case 'A':  tebPrms.ifAddr     = optarg;             break;
      case '1':  tebPrms.core[0]    = atoi(optarg);       break;
      case '2':  tebPrms.core[1]    = atoi(optarg);       break;
      case 'u':  tebPrms.alias      = optarg;             break;
      case 'v':  ++tebPrms.verbose;
                 ++mebPrms.verbose;                       break;
      case '?':
      case 'h':
        usage(argv[0], (char*)"Test DRP event contributor", tebPrms);
        return 1;
    }
  }

  if ( (mebPrms.partition = tebPrms.partition) == NO_PARTITION)
  {
    fprintf(stderr, "Missing '%s' parameter\n", "-p <Partition number>");
    return 1;
  }
  if (collSrv.empty())
  {
    fprintf(stderr, "Missing '%s' parameter\n", "-C <Collection server>");
    return 1;
  }

  // Revisit: Fix MAX_ENTRIES to equal duration?
  if (MAX_ENTRIES > BATCH_DURATION)
  {
    fprintf(stderr, "More batch entries (%u) requested than definable "
            "in the batch duration (%lu)\n",
            MAX_ENTRIES, BATCH_DURATION);
    abort();
  }

  struct sigaction sigAction;

  sigAction.sa_handler = sigHandler;
  sigAction.sa_flags   = SA_RESTART;
  sigemptyset(&sigAction.sa_mask);
  if (sigaction(SIGINT, &sigAction, &lIntAction) > 0)
    fprintf(stderr, "Failed to set up ^C handler\n");

  prometheus::Exposer exposer{"0.0.0.0:9200", "/metrics", 1};
  auto exporter = std::make_shared<MetricExporter>();

  CtrbApp app(collSrv, tebPrms, mebPrms, exporter);

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
