#include "EbAppBase.hh"

#include "BatchManager.hh"
#include "EbEvent.hh"

#include "EbLfClient.hh"
#include "EbLfServer.hh"

#include "utilities.hh"

#include "psdaq/trigger/Trigger.hh"
#include "psdaq/trigger/utilities.hh"
#include "psdaq/service/kwargs.hh"
#include "psdaq/service/MetricExporter.hh"
#include "psdaq/service/Collection.hh"
#include "psdaq/service/Dl.hh"
#include "psalg/utils/SysLog.hh"
#include "xtcdata/xtc/Dgram.hh"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <stdio.h>
#include <unistd.h>                     // For getopt(), gethostname()
#include <cstring>
#include <climits>                      // For HOST_NAME_MAX
#include <csignal>
#include <bitset>
#include <atomic>
#include <vector>
#include <cassert>
#include <iostream>
#include <sstream>
#include <exception>
#include <algorithm>                    // For std::fill()
#include <Python.h>

#include "rapidjson/document.h"

using namespace rapidjson;
using namespace XtcData;
using namespace Pds;
using namespace Pds::Trg;

using json     = nlohmann::json;
using logging  = psalg::SysLog;
using string_t = std::string;

static const int CORE_0 = -1;           // devXXX: 18, devXX:  7, accXX:  9
static const int CORE_1 = -1;           // devXXX: 19, devXX: 19, accXX: 21

static struct sigaction      lIntAction;
static volatile sig_atomic_t lRunning = 1;

void sigHandler( int signal )
{
  static unsigned callCount(0);

  if (callCount == 0)
  {
    logging::info("Shutting down");

    lRunning = 0;
  }

  if (callCount++)
  {
    logging::critical("Aborting on 2nd ^C");

    sigaction(signal, &lIntAction, NULL);
    raise(signal);
  }
}


namespace Pds {
  namespace Eb {

    using MetricExporter_t = std::shared_ptr<MetricExporter>;

    class Teb : public EbAppBase
    {
    public:
      Teb(const EbParams& prms, const MetricExporter_t& exporter);
    public:
      int      resetCounters();
      int      startConnection(std::string& tebPort, std::string& mrqPort);
      int      connect();
      int      configure(Trigger* object, unsigned prescale);
      int      beginrun ();
      void     unconfigure();
      void     disconnect();
      void     shutdown();
      void     run();
    public:                         // For EventBuilder
      virtual
      void     flush() override;
      virtual
      void     process(EbEvent* event) override;
    private:
      void     _tryPost(const EbDgram* dg, uint64_t dsts);
      void     _post(const EbDgram* start, const EbDgram* end);
      uint64_t _receivers(unsigned rogs) const;
    private:
      std::vector<EbLfCltLink*> _l3Links;
      EbLfServer                _mrqTransport;
      std::vector<EbLfSvrLink*> _mrqLinks;
      BatchManager              _batMan;
      unsigned                  _id;
      const EbDgram*            _batchStart;
      const EbDgram*            _batchEnd;
      uint64_t                  _resultDsts;
    private:
      EbAppBase::u64arr_t       _rcvrs;
      //uint64_t                  _trimmed;
      Trigger*                  _trigger;
      unsigned                  _prescale;
    private:
      unsigned                  _wrtCounter;
      uint64_t                  _pidPrv;
    private:
      uint64_t                  _eventCount;
      uint64_t                  _splitCount;
      uint64_t                  _batchCount;
      uint64_t                  _writeCount;
      uint64_t                  _monitorCount;
      uint64_t                  _prescaleCount;
    private:
      const EbParams&           _prms;
      EbLfClient                _l3Transport;
    };
  };
};


using namespace Pds::Eb;

Teb::Teb(const EbParams&         prms,
         const MetricExporter_t& exporter) :
  EbAppBase     (prms, exporter, "TEB", BATCH_DURATION, MAX_ENTRIES, MAX_BATCHES),
  _mrqTransport (prms.verbose, prms.kwargs),
  _id           (-1),
  _batchStart   (nullptr),
  _batchEnd     (nullptr),
  _resultDsts   (0),
  //_trimmed      (0),
  _trigger      (nullptr),
  _pidPrv       (0),
  _eventCount   (0),
  _splitCount   (0),
  _batchCount   (0),
  _writeCount   (0),
  _monitorCount (0),
  _prescaleCount(0),
  _prms         (prms),
  _l3Transport  (prms.verbose, prms.kwargs)
{
  std::map<std::string, std::string> labels{{"instrument", prms.instrument},
                                            {"partition", std::to_string(prms.partition)},
                                            {"detname", prms.alias},
                                            {"alias", prms.alias}};
  exporter->add("TEB_EvtRt",  labels, MetricType::Rate,    [&](){ return _eventCount;             });
  exporter->add("TEB_EvtCt",  labels, MetricType::Counter, [&](){ return _eventCount;             });
  exporter->add("TEB_SpltCt", labels, MetricType::Counter, [&](){ return _splitCount;             });
  exporter->add("TEB_BatCt",  labels, MetricType::Counter, [&](){ return _batchCount;             }); // Outbound
  exporter->add("TEB_BtAlCt", labels, MetricType::Counter, [&](){ return _batMan.batchAllocCnt(); });
  exporter->add("TEB_BtFrCt", labels, MetricType::Counter, [&](){ return _batMan.batchFreeCnt();  });
  exporter->add("TEB_BtWtg",  labels, MetricType::Gauge,   [&](){ return _batMan.batchWaiting();  });
  exporter->add("TEB_TxPdg",  labels, MetricType::Gauge,   [&](){ return _l3Transport.pending();  });
  exporter->add("TEB_WrtRt",  labels, MetricType::Rate,    [&](){ return _writeCount;             });
  exporter->add("TEB_WrtCt",  labels, MetricType::Counter, [&](){ return _writeCount;             });
  exporter->add("TEB_MonRt",  labels, MetricType::Rate,    [&](){ return _monitorCount;           });
  exporter->add("TEB_MonCt",  labels, MetricType::Counter, [&](){ return _monitorCount;           });
  exporter->add("TEB_PsclCt", labels, MetricType::Counter, [&](){ return _prescaleCount;          });
}

int Teb::resetCounters()
{
  EbAppBase::resetCounters();

  _eventCount    = 0;
  _splitCount    = 0;
  _batchCount    = 0;
  _writeCount    = 0;
  _monitorCount  = 0;
  _prescaleCount = 0;

  return 0;
}

void Teb::shutdown()
{
  if (!_l3Links.empty())                // Avoid shutting down if already done
  {
    unconfigure();
    disconnect();

    _mrqTransport.shutdown();
  }
}

void Teb::disconnect()
{
  for (auto link : _mrqLinks)  _mrqTransport.disconnect(link);
  _mrqLinks.clear();

  for (auto link : _l3Links)  _l3Transport.disconnect(link);
  _l3Links.clear();

  EbAppBase::disconnect();

  _id = -1;
  _rcvrs.fill(0);
}

void Teb::unconfigure()
{
  _batMan.dump();
  _batMan.shutdown();

  EbAppBase::unconfigure();
}

int Teb::startConnection(std::string& tebPort,
                         std::string& mrqPort)
{
  int rc = EbAppBase::startConnection(_prms.ifAddr, tebPort, MAX_DRPS);
  if (rc)  return rc;

  rc = linksStart(_mrqTransport, _prms.ifAddr, mrqPort, MAX_MRQS, "MRQ");
  if (rc)  return rc;

  return 0;
}

int Teb::connect()
{
  _l3Links .resize(_prms.addrs.size());
  _mrqLinks.resize(_prms.numMrqs);
  _id      = _prms.id;
  _rcvrs   = _prms.receivers;

  // Make a guess at the size of the Input entries
  size_t inpSizeGuess = sizeof(EbDgram) + 2  * sizeof(uint32_t);

  int rc = EbAppBase::connect(_prms, inpSizeGuess);
  if (rc)  return rc;

  rc = linksConnect(_l3Transport, _l3Links, _prms.addrs, _prms.ports, "DRP");
  if (rc)  return rc;
  rc = linksConnect(_mrqTransport, _mrqLinks, "MRQ");
  if (rc)  return rc;

  if (!_batMan.batchRegion())           // No need to guess again
  {
    // Make a guess at the size of the Result entries
    size_t maxResultSizeGuess = sizeof(EbDgram) + 2 * sizeof(uint32_t);
    _batMan.initialize(maxResultSizeGuess, true); // TEB always batches
  }

  void*  region  = _batMan.batchRegion();
  size_t regSize = _batMan.batchRegionSize();

  //printf("*** TEB::connect: region %p, regSize %zu\n", region, regSize);
  for (auto link : _l3Links)
  {
    rc = link->setupMr(region, regSize);
    if (rc)  return rc;
  }

  return 0;
}

int Teb::configure(Trigger* object,
                   unsigned prescale)
{
  _trigger    = object;
  _prescale   = prescale - 1;           // Be zero based
  _wrtCounter = _prescale;              // Reset prescale counter

  int rc = EbAppBase::configure(_prms);
  if (rc)  return rc;

  // maxResultSize becomes known during Configure, so reinitialize BatchManager now
  _batMan.initialize(_prms.maxResultSize, true); // TEB always batches

  void*  region  = _batMan.batchRegion();
  size_t regSize = _batMan.batchRegionSize();

  //printf("*** TEB::cfg: region %p, regSize %zu\n", region, regSize);
  rc = linksConfigure(_l3Links, _id, region, regSize, "DRP");
  if (rc)  return rc;
  rc = linksConfigure(_mrqLinks, _id, "MRQ");
  if (rc)  return rc;

  return 0;
}

int Teb::beginrun()
{
  _eventCount    = 0;
  _writeCount    = 0;
  _monitorCount  = 0;
  return 0;
}

void Teb::run()
{
  logging::info("TEB thread started");

  int rc = pinThread(pthread_self(), _prms.core[0]);
  if (rc != 0)
  {
    logging::debug("%s:\n  Error from pinThread:\n  %s",
                   __PRETTY_FUNCTION__, strerror(rc));
  }

  _batchStart    = nullptr;
  _batchEnd      = nullptr;
  _resultDsts    = 0;
  //_trimmed       = 0;
  _eventCount    = 0;
  _splitCount    = 0;
  _batchCount    = 0;
  _writeCount    = 0;
  _monitorCount  = 0;
  _prescaleCount = 0;

  while (lRunning)
  {
    if (EbAppBase::process() < 0)
    {
      if (checkEQ() == -FI_ENOTCONN)
      {
        logging::error("TEB thread lost connection with a DRP");
        break;
      }
    }
  }

  logging::info("TEB thread finished");
}

void Teb::process(EbEvent* event)
{
  // Accumulate output datagrams (result) from the event builder into a batch
  // datagram.  When the batch completes, post it to the contributors.

  if (unlikely(_prms.verbose >= VL_DETAILED))
  {
    static unsigned cnt = 0;
    printf("Teb::process event dump:\n");
    event->dump(++cnt);
  }

  const EbDgram* dgram = event->creator();
  if (!(dgram->readoutGroups() & (1 << _prms.partition)))
  {
    // The common readout group keeps events and batches in pulse ID order
    logging::error("%s:\n  Event %014lx, env %08x is missing the common readout group %u",
                   __PRETTY_FUNCTION__, dgram->pulseId(), dgram->env, _prms.partition);
    // Revisit: Should this be fatal?
  }

  uint64_t pid = dgram->pulseId();
  if (!(pid > _pidPrv))
  {
    if (event->remaining())             // I.e., this event was fixed up
    {
      // This can happen only for a split event (I think), which was fixed up and
      // posted earlier, so return to dismiss this counterpart and not post it
      ++_splitCount;
      return;
    }

    event->damage(Damage::OutOfOrder);

    logging::error("%s:\n  Pulse ID did not advance: %014lx vs %014lx\n",
                   __PRETTY_FUNCTION__, pid, _pidPrv);
    // Revisit: fatal?  throw "Pulse ID did not advance";
  }
  _pidPrv = pid;

  ++_eventCount;

  // "Selected" EBs respond with a Result, others simply acknowledge
  if (ImmData::rsp(ImmData::flg(event->parameter())) == ImmData::Response)
  {
    Batch*       batch = _batMan.fetch(pid);
    ResultDgram* rdg   = new(batch->allocate()) ResultDgram(*dgram, _id);

    rdg->xtc.damage.increase(event->damage().value());

    if (rdg->isEvent())
    {
      // Present event contributions to "user" code for building a result datagram
      _trigger->event(event->begin(), event->end(), *rdg); // Consume

      unsigned line = 0;                // Revisit: For future expansion

      // Handle prescale
      if (!rdg->persist(line) && !_wrtCounter--)
      {
        _prescaleCount++;

        rdg->prescale(line, true);
        _wrtCounter = _prescale;
      }

      if (rdg->persist())  _writeCount++;
      if (rdg->monitor())
      {
        uint64_t data;
        int      rc = _mrqTransport.poll(&data);
        if (rc > 0)
        {
          _monitorCount++;

          rdg->monBufNo(data);
        }
        else
          rdg->monitor(line, false);
      }
    }

    // Avoid sending Results to contributors that failed to supply Input
    uint64_t dsts = _receivers(dgram->readoutGroups()) & ~event->remaining();

    if (unlikely(_prms.verbose >= VL_EVENT)) // || rdg->monitor()))
    {
      const char* svc = TransitionId::name(rdg->service());
      uint64_t    pid = rdg->pulseId();
      unsigned    idx = Batch::index(pid);
      unsigned    ctl = rdg->control();
      size_t      sz  = sizeof(rdg) + rdg->xtc.sizeofPayload();
      unsigned    src = rdg->xtc.src.value();
      unsigned    env = rdg->env;
      uint32_t*   pld = reinterpret_cast<uint32_t*>(rdg->xtc.payload());
      printf("TEB processed %15s result [%8u] @ "
             "%16p, ctl %02x, pid %014lx, env %08x, sz %6zd, src %2u, dsts %016lx, res [%08x, %08x]\n",
             svc, idx, rdg, ctl, pid, env, sz, src, dsts, pld[0], pld[1]);
    }

    _tryPost(rdg, dsts);
  }
  else
  {
    // Flush whatever batch there is
    if (_batchStart)
    {
      const EbDgram*      start   = _batchStart;
      const EbDgram*      end     = _batchEnd;
      TransitionId::Value svc     = dgram->service();
      bool                flush   = !((svc == TransitionId::L1Accept) ||
                                      (svc == TransitionId::SlowUpdate));
      bool                expired = _batMan.expired(pid, start->pulseId());

      if (expired || flush)
      {
        if (!end)  end = start;

        _post(start, end);
        // Only transitions come through here

        // Start a new batch
        _batchStart = nullptr;
        _batchEnd   = nullptr;
        _resultDsts = 0;
      }
    }

    if (unlikely(_prms.verbose >= VL_EVENT)) // || rdg->monitor()))
    {
      const char* svc = TransitionId::name(dgram->service());
      unsigned    idx = Batch::index(pid);
      unsigned    ctl = dgram->control();
      size_t      sz  = sizeof(dgram) + dgram->xtc.sizeofPayload();
      unsigned    src = dgram->xtc.src.value();
      unsigned    env = dgram->env;
      printf("TEB processed %15s ACK    [%8u] @ "
             "%16p, ctl %02x, pid %014lx, env %08x, sz %6zd, src %2u, data %08x\n",
             svc, idx, dgram, ctl, pid, env, sz, src, event->parameter());
    }

    // Make the transition buffer available to the contributor again
    post(event->begin(), event->end());
  }
}

// Called by EB  on timeout when it is empty of events
// to flush out any in-progress batch
void Teb::flush()
{
  const EbDgram* start = _batchStart;
  const EbDgram* end   = _batchEnd;

  //printf("TEB::flush: start %p, end %p\n", start, end);

  if (start)
  {
    //printf("TEB::flush:    posting %014lx - %014lx\n", start->pulseId(), end->pulseId());

    _post(start, end);

    _batchStart = nullptr;
    _batchEnd   = nullptr;
    _resultDsts = 0;
  }
}

void Teb::_tryPost(const EbDgram* dgram, uint64_t dsts)
{
  //printf("tryPost: pid %014lx, batchStart %014lx\n", dgram->pulseId(), _batchStart ? _batchStart->pulseId() : 0ul);

  // The batch start is the first dgram seen
  if (!_batchStart)  _batchStart = dgram;

  const EbDgram*      start   = _batchStart;
  const EbDgram*      end     = _batchEnd;
  TransitionId::Value svc     = dgram->service();
  bool                flush   = !((svc == TransitionId::L1Accept) ||
                                  (svc == TransitionId::SlowUpdate));
  bool                expired = _batMan.expired(dgram->pulseId(), start->pulseId());

  //printf("tryPost: flush %d, expired %d\n", flush, expired);

  if (expired || flush)
  {
    if (expired)
    {
      if (!end)
      {
        end          = dgram;
        _resultDsts |= dsts;
      }

      //printf("tryPost: e||f  posting %014lx - %014lx\n", start->pulseId(), end->pulseId());

      _post(start, end);

      // Start a new batch
      _batchStart = end == dgram ? nullptr : dgram;
      _batchEnd   = end == dgram ? nullptr : dgram;
      _resultDsts = end == dgram ? 0       : dsts;
      start       = _batchStart;

      //printf("tryPost: e||f  batchStart %014lx, batchEnd %014lx\n",
      //       _batchStart ? _batchStart->pulseId() : 0ul,
      //       _batchEnd   ? _batchEnd->pulseId()   : 0ul);
    }

    if (flush && start)     // Post the batch + transition if it wasn't just done
    {
      _resultDsts |= dsts;

      //printf("tryPost: f     posting %014lx - %014lx\n", start->pulseId(), dgram->pulseId());

      _post(start, dgram);

      // Start a new batch
      _batchStart = nullptr;
      _batchEnd   = nullptr;
      _resultDsts = 0;
    }
  }
  else
  {
    _batchEnd    = dgram;   // The batch end is the one before the current dgram
    _resultDsts |= dsts;

      //printf("tryPost: else  batchStart %014lx, batchEnd %014lx\n",
      //       _batchStart ? _batchStart->pulseId() : 0ul,
      //       _batchEnd   ? _batchEnd->pulseId()   : 0ul);
  }
}

void Teb::_post(const EbDgram* start, const EbDgram* end)
{
  uint64_t pid    = start->pulseId();
  uint32_t idx    = Batch::index(pid);
  size_t   extent = (reinterpret_cast<const char*>(end) -
                     reinterpret_cast<const char*>(start)) + _prms.maxResultSize;
  unsigned offset = idx * _prms.maxResultSize;
  uint64_t data   = ImmData::value(ImmData::Buffer, _id, idx);
  uint64_t destns = _resultDsts; // & ~_trimmed;

  end->setEOL();                        // Terminate the batch

  if (unlikely(_prms.verbose >= VL_BATCH))
  {
    printf("TEB posts          %9lu result  [%8u] @ "
           "%16p,         pid %014lx,               sz %6zd, dst %016lx\n",
           _batchCount, idx, start, pid, extent, destns);
  }

  while (destns)
  {
    unsigned     dst  = __builtin_ffsl(destns) - 1;
    EbLfCltLink* link = _l3Links[dst];

    destns &= ~(1ul << dst);

    if (int rc = link->post(start, extent, offset, data) < 0)
    {
      if (rc != -FI_ETIMEDOUT)  break;  // Revisit: Right thing to do?

      // If we were to trim, here's how to do it.  For now, we don't.
      //static unsigned retries = 0;
      //trim(dst);
      //if (retries++ == 5)  { _trimmed |= 1ul << dst; retries = 0; }
      //printf("%s:  link->post() to %u returned %d, trimmed = %016lx\n",
      //       __PRETTY_FUNCTION__, dst, rc, _trimmed);
    }
  }

  ++_batchCount;

  // Revisit: The following deallocation constitutes a race with the posts to
  // the transport above as the batch's memory cannot be allowed to be reused
  // for a subsequent batch before the transmit completes.  Waiting for
  // completion here would impact performance.  Since there are many batches,
  // and only one is active at a time, a previous batch will have completed
  // transmitting before the next one starts (or the subsequent transmit won't
  // start), thus making it safe to "pre-delete" it here.
  _batMan.release(pid);
}

uint64_t Teb::_receivers(unsigned groups) const
{
  // This method is called when the event is processed, which happens when the
  // event builder has built the event.  The supplied contribution contains
  // information from the L1 trigger that identifies which readout groups were
  // involved.  This routine can thus look up the list of receivers expecting
  // results from the event for each of the readout groups and logically OR
  // them together to provide the overall receiver list.  The list of receivers
  // in each readout group desiring event results is provided at configuration
  // time.

  uint64_t receivers = 0;

  while (groups)
  {
    unsigned group = __builtin_ffs(groups) - 1;
    groups &= ~(1 << group);

    receivers |= _rcvrs[group];
  }
  return receivers;
}


static std::string getHostname()
{
  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);
  return std::string(hostname);
}

class TebApp : public CollectionApp
{
public:
  TebApp(const std::string& collSrv, EbParams&);
  ~TebApp();
public:                                 // For CollectionApp
  json connectionInfo() override;
  void handleConnect(const json& msg) override;
  void handleDisconnect(const json& msg) override;
  void handlePhase1(const json& msg) override;
  void handleReset(const json& msg) override;
private:
  std::string
       _error(const json& msg, const std::string& errorMsg);
  int  _configure(const json& msg);
  void _unconfigure();
  int  _parseConnectionParams(const json& msg);
  void _printParams(const EbParams& prms, unsigned groups) const;
  void _printGroups(unsigned groups, const EbAppBase::u64arr_t& array) const;
  void _buildContract(const Document& top);
private:
  EbParams&                            _prms;
  const bool                           _ebPortEph;
  const bool                           _mrqPortEph;
  std::unique_ptr<prometheus::Exposer> _exposer;
  std::shared_ptr<MetricExporter>      _exporter;
  std::unique_ptr<Teb>                 _teb;
  std::thread                          _appThread;
  json                                 _connectMsg;
  Trg::Factory<Trg::Trigger>           _factory;
  uint16_t                             _groups;
  bool                                 _unconfigFlag;
};

TebApp::TebApp(const std::string& collSrv,
               EbParams&          prms) :
  CollectionApp(collSrv, prms.partition, "teb", prms.alias),
  _prms        (prms),
  _ebPortEph   (prms.ebPort.empty()),
  _mrqPortEph  (prms.mrqPort.empty()),
  _exposer     (Pds::createExposer(prms.prometheusDir, getHostname())),
  _exporter    (std::make_shared<MetricExporter>()),
  _teb         (std::make_unique<Teb>(_prms, _exporter)),
  _unconfigFlag(false)
{
  Py_Initialize();

  if (_exposer)
  {
    _exposer->RegisterCollectable(_exporter);
  }

  logging::info("Ready for transitions");
}

TebApp::~TebApp()
{
  // Try to take things down gracefully when an exception takes us off the
  // normal path so that the most chance is given for prints to show up
  handleReset(json({}));

  Py_Finalize();
}

std::string TebApp::_error(const json&        msg,
                           const std::string& errorMsg)
{
  json body = json({});
  const std::string& key = msg["header"]["key"];
  body["err_info"] = errorMsg;
  logging::error("%s", errorMsg.c_str());
  reply(createMsg(key, msg["header"]["msg_id"], getId(), body));
  return errorMsg;
}

json TebApp::connectionInfo()
{
  // Allow the default NIC choice to be overridden
  if (_prms.ifAddr.empty())
  {
    _prms.ifAddr = _prms.kwargs.find("ep_domain") != _prms.kwargs.end()
                 ? getNicIp(_prms.kwargs["ep_domain"])
                 : getNicIp(_prms.kwargs["forceEnet"] == "yes");
  }
  logging::debug("nic ip  %s", _prms.ifAddr.c_str());

  // If port is not user specified, reset the previously allocated port number
  if (_ebPortEph)   _prms.ebPort.clear();
  if (_mrqPortEph)  _prms.mrqPort.clear();

  int rc = _teb->startConnection(_prms.ebPort, _prms.mrqPort);
  if (rc)  throw "Error starting connection";

  json body = {{"connect_info", {{"nic_ip",   _prms.ifAddr},
                                 {"teb_port", _prms.ebPort},
                                 {"mrq_port", _prms.mrqPort}}}};
  return body;
}

void TebApp::handleConnect(const json& msg)
{
  // Save a copy of the json so we can use it to connect to
  // the config database on configure
  _connectMsg = msg;

  json body = json({});
  int  rc   = _parseConnectionParams(msg["body"]);
  if (rc)
  {
    _error(msg, "Error parsing connection parameters");
    return;
  }

  _teb->resetCounters();

  rc = _teb->connect();
  if (rc)
  {
    _error(msg, "Error in TEB connect()");
    return;
  }

  // Reply to collection with transition status
  reply(createMsg("connect", msg["header"]["msg_id"], getId(), body));
}

void TebApp::_buildContract(const Document& top)
{
  const json& body = _connectMsg["body"];

  bool buildAll = top.HasMember("buildAll") && top["buildAll"].GetInt()==1;
  _prms.contractors.fill(0);

  for (auto it : body["drp"].items())
  {
    unsigned    drpId   = it.value()["drp_id"];
    std::string alias   = it.value()["proc_info"]["alias"];
    size_t      found   = alias.rfind('_');
    std::string detName = alias.substr(0, found);

    if (buildAll || top.HasMember(detName.c_str()))
    {
      auto group = unsigned(it.value()["det_info"]["readout"]);
      _prms.contractors[group] |= 1ul << drpId;
    }
  }
}

int TebApp::_configure(const json& msg)
{
  int               rc = 0;
  Document          top;
  const std::string configAlias(msg["body"]["config_alias"]);
  const std::string triggerConfig(msg["body"]["trigger_config"]);

  // In the following, _0 is added in prints to show the default segment number
  logging::info("Fetching trigger info from ConfigDb/%s/%s_0",
                configAlias.c_str(), triggerConfig.c_str());

  if (Pds::Trg::fetchDocument(_connectMsg.dump(), configAlias, triggerConfig, top))
  {
    logging::error("%s:\n  Document '%s_0' not found in ConfigDb",
                   __PRETTY_FUNCTION__, triggerConfig.c_str());
    return -1;
  }

  if (!triggerConfig.empty())  _buildContract(top);

  const std::string symbol("create_consumer");
  Trigger* trigger = _factory.create(top, triggerConfig, symbol);
  if (!trigger)
  {
    logging::error("%s:\n  Failed to create Trigger",
                   __PRETTY_FUNCTION__);
    return -1;
  }
  _prms.maxResultSize = trigger->size();

  if (trigger->configure(_connectMsg, top))
  {
    logging::error("%s:\n  Failed to configure Trigger",
                   __PRETTY_FUNCTION__);
    return -1;
  }

# define _FETCH(key, item)                                              \
  if (top.HasMember(key))  item = top[key].GetUint();                   \
  else { logging::error("%s:\n  Key '%s' not found in Document %s",     \
                        __PRETTY_FUNCTION__, key, triggerConfig.c_str()); \
         rc = -1; }

  unsigned prescale;  _FETCH("prescale", prescale);

# undef _FETCH

  rc = _teb->configure(trigger, prescale);
  if (rc)  logging::error("%s:\n  Failed to configure TEB",
                          __PRETTY_FUNCTION__);

  return rc;
}

void TebApp::_unconfigure()
{
  // Shut down the previously running instance, if any
  lRunning = 0;
  if (_appThread.joinable())  _appThread.join();

  _teb->unconfigure();
}

void TebApp::handlePhase1(const json& msg)
{
  json        body = json({});
  std::string key  = msg["header"]["key"];

  if (key == "configure")
  {
    // Handle a "queued" Unconfigure, if any
    if (_unconfigFlag)
    {
      _unconfigure();
      _unconfigFlag = false;
    }

    int rc = _configure(msg);
    if (rc)
    {
      _error(msg, "Phase 1 error: Failed to " + key);
      return;
    }

    _printParams(_prms, _groups);

    lRunning = 1;

    _appThread = std::thread(&Teb::run, std::ref(*_teb));
  }
  else if (key == "unconfigure")
  {
    // "Queue" unconfiguration until after phase 2 has completed
    _unconfigFlag = true;
  }
  else if (key == "beginrun")
  {
    if (_teb->beginrun())
    {
      _error(msg, "Phase 1 error: Failed to " + key);
      return;
    }
  }

  // Reply to collection with transition status
  reply(createMsg(key, msg["header"]["msg_id"], getId(), body));
}

void TebApp::handleDisconnect(const json& msg)
{
  // Carry out the queued Unconfigure, if there was one
  if (_unconfigFlag)
  {
    _unconfigure();
    _unconfigFlag = false;
  }

  _teb->disconnect();

  // Reply to collection with transition status
  json body = json({});
  reply(createMsg("disconnect", msg["header"]["msg_id"], getId(), body));
}

void TebApp::handleReset(const json& msg)
{
  lRunning = 0;
  if (_appThread.joinable())  _appThread.join();

  _teb->shutdown();
}

int TebApp::_parseConnectionParams(const json& body)
{
  std::string id = std::to_string(getId());
  _prms.id = body["teb"][id]["teb_id"];
  if (_prms.id >= MAX_TEBS)
  {
    logging::error("TEB ID %d is out of range 0 - %u", _prms.id, MAX_TEBS - 1);
    return 1;
  }

  _prms.contributors = 0;
  _prms.addrs.clear();
  _prms.ports.clear();

  _prms.contractors.fill(0);
  _prms.receivers.fill(0);
  _groups = 0;

  if (body.find("drp") == body.end())
  {
    logging::error("Missing required DRP specs");
    return 1;
  }

  for (auto it : body["drp"].items())
  {
    unsigned    drpId   = it.value()["drp_id"];
    if (drpId > MAX_DRPS - 1)
    {
      logging::error("DRP ID %d is out of range 0 - %u", drpId, MAX_DRPS - 1);
      return 1;
    }
    _prms.contributors |= 1ul << drpId;

    _prms.addrs.push_back(it.value()["connect_info"]["nic_ip"]);
    _prms.ports.push_back(it.value()["connect_info"]["drp_port"]);

    auto group = unsigned(it.value()["det_info"]["readout"]);
    if (group > NUM_READOUT_GROUPS - 1)
    {
      logging::error("Readout group %u is out of range 0 - %u", group, NUM_READOUT_GROUPS - 1);
      return 1;
    }
    _prms.contractors[group] |= 1ul << drpId; // Possibly overridden during Configure
    _prms.receivers[group]   |= 1ul << drpId; // All contributors receive results
    _groups |= 1 << group;
  }

  auto& vec =_prms.maxTrSize;
  vec.resize(body["drp"].size());
  std::fill(vec.begin(), vec.end(), sizeof(EbDgram)); // Same for all contributors

  _prms.numMrqs = 0;
  if (body.find("meb") != body.end())
  {
    for (auto it : body["meb"].items())
    {
      _prms.numMrqs++;
    }
    if (_prms.numMrqs > MAX_MRQS)
    {
      logging::error("More monitor requestors found (%u) than supportable %u",
                     _prms.numMrqs, MAX_MRQS);
      return 1;
    }
  }

  return 0;
}

void TebApp::_printGroups(unsigned groups, const EbAppBase::u64arr_t& array) const
{
  while (groups)
  {
    unsigned group = __builtin_ffs(groups) - 1;
    groups &= ~(1 << group);

    printf("%u: 0x%016lx  ", group, array[group]);
  }
  printf("\n");
}

void TebApp::_printParams(const EbParams& prms, unsigned groups) const
{
  printf("Parameters of TEB ID %d (%s:%s):\n",                   prms.id,
                                                                 prms.ifAddr.c_str(), prms.ebPort.c_str());
  printf("  Thread core numbers:          %d, %d\n",             prms.core[0], prms.core[1]);
  printf("  Partition:                    %u\n",                 prms.partition);
  printf("  Bit list of contributors:     0x%016lx, cnt: %zd\n", prms.contributors,
                                                                 std::bitset<64>(prms.contributors).count());
  printf("  Readout group contractors:    ");                    _printGroups(groups, prms.contractors);
  printf("  Readout group receivers:      ");                    _printGroups(groups, prms.receivers);
  printf("  Number of MEB requestors:     %u\n",                 prms.numMrqs);
  printf("  Batch duration:               0x%014lx = %lu uS\n",  BATCH_DURATION, BATCH_DURATION);
  printf("  Batch pool depth:             0x%08x = %u\n",        MAX_BATCHES, MAX_BATCHES);
  printf("  Max # of entries / batch:     0x%08x = %u\n",        MAX_ENTRIES, MAX_ENTRIES);
  printf("  # of contrib. buffers:        0x%08x = %u\n",        MAX_LATENCY, MAX_LATENCY);
  printf("  Max result     EbDgram size:  0x%08zx = %zd\n",      prms.maxResultSize, prms.maxResultSize);
  printf("  Max transition EbDgram size:  0x%08zx = %zd\n",      prms.maxTrSize[0], prms.maxTrSize[0]);
  printf("\n");
}


static
void usage(char *name, char *desc, const EbParams& prms)
{
  fprintf(stderr, "Usage:\n");
  fprintf(stderr, "  %s [OPTIONS]\n", name);

  if (desc)
    fprintf(stderr, "\n%s\n", desc);

  fprintf(stderr, "\nOptions:\n");

  fprintf(stderr, " %-23s %s (default: %s)\n",        "-A <interface_addr>",
          "IP address of the interface to use",       "libfabric's 'best' choice");
  fprintf(stderr, " %-23s %s (default: %s)\n",        "-E <TEB server port>",
          "Port served to Contributors for Inputs",   "dynamically assigned");
  fprintf(stderr, " %-23s %s (default: %s)\n",        "-R <MRQ server port>",
          "Network port for Mon requestors",          "dynamically assigned");

  fprintf(stderr, " %-23s %s (required)\n",           "-C <address>",
          "Collection server");
  fprintf(stderr, " %-23s %s (required)\n",           "-p <partition number>",
          "Partition number");
  fprintf(stderr, " %-23s %s\n",                      "-P <instrument>",
          "Instrument name");
  fprintf(stderr, " %-23s %s (required)\n",           "-u <alias>",
          "Alias for teb process");
  fprintf(stderr, " %-23s %s\n",                      "-M <directory>",
          "Prometheus config file directory");
  fprintf(stderr, " %-23s %s\n",                      "-k <key=value>[, ...]",
          "Keyword arguments");
  fprintf(stderr, " %-23s %s (default: %u)\n",        "-1 <core>",
          "Core number for pinning App thread to",    CORE_0);
  fprintf(stderr, " %-23s %s (default: %u)\n",        "-2 <core>",
          "Core number for pinning other threads to", CORE_1);

  fprintf(stderr, " %-23s %s\n", "-v", "enable debugging output (repeat for increased detail)");
  fprintf(stderr, " %-23s %s\n", "-h", "display this help output");
}


int main(int argc, char **argv)
{
  const unsigned NO_PARTITION = unsigned(-1u);
  int            op           = 0;
  std::string    collSrv;
  EbParams       prms;
  std::string    kwargs_str;

  prms.instrument = {};
  prms.partition = NO_PARTITION;
  prms.core[0]   = CORE_0;
  prms.core[1]   = CORE_1;
  prms.verbose   = 0;

  while ((op = getopt(argc, argv, "C:p:P:A:E:R:1:2:u:M:k:h?v")) != -1)
  {
    switch (op)
    {
      case 'C':  collSrv            = optarg;                       break;
      case 'p':  prms.partition     = std::stoi(optarg);            break;
      case 'P':  prms.instrument    = optarg;                       break;
      case 'A':  prms.ifAddr        = optarg;                       break;
      case 'E':  prms.ebPort        = optarg;                       break;
      case 'R':  prms.mrqPort       = optarg;                       break;
      case '1':  prms.core[0]       = atoi(optarg);                 break;
      case '2':  prms.core[1]       = atoi(optarg);                 break;
      case 'u':  prms.alias         = optarg;                       break;
      case 'M':  prms.prometheusDir = optarg;                       break;
      case 'k':  kwargs_str         = kwargs_str.empty()
                                    ? optarg
                                    : kwargs_str + ", " + optarg;   break;
      case 'v':  ++prms.verbose;                                    break;
      case '?':
      case 'h':
      default:
        usage(argv[0], (char*)"Trigger Event Builder application", prms);
        return 1;
    }
  }

  logging::init(prms.instrument.c_str(), prms.verbose ? LOG_DEBUG : LOG_INFO);
  logging::info("logging configured");

  if (prms.instrument.empty())
  {
    logging::warning("-P: instrument name is missing");
  }
  if (prms.partition == NO_PARTITION)
  {
    logging::critical("-p: partition number is mandatory");
    return 1;
  }
  if (collSrv.empty())
  {
    logging::critical("-C: collection server is mandatory");
    return 1;
  }
  if (prms.alias.empty()) {
    logging::critical("-u: alias is mandatory");
    return 1;
  }

  get_kwargs(kwargs_str, prms.kwargs);

  struct sigaction sigAction;

  sigAction.sa_handler = sigHandler;
  sigAction.sa_flags   = SA_RESTART;
  sigemptyset(&sigAction.sa_mask);
  if (sigaction(SIGINT, &sigAction, &lIntAction) > 0)
    logging::error("Failed to set up ^C handler");

  // Event builder sorts contributions into a time ordered list
  // Then calls the user's process() with complete events to build the result datagram
  // Post completed result batches to the outlet

  // Iterate over contributions in the batch
  // Event build them according to their trigger group

  try
  {
    TebApp app(collSrv, prms);

    app.run();

    return 0;
  }
  catch (std::exception& e)  { logging::critical("%s", e.what()); }
  catch (std::string& e)     { logging::critical("%s", e.c_str()); }
  catch (char const* e)      { logging::critical("%s", e); }
  catch (...)                { logging::critical("Default exception"); }

  return EXIT_FAILURE;
}
