#include "EbAppBase.hh"

#include "BatchManager.hh"
#include "EbEvent.hh"

#include "EbLfClient.hh"
#include "EbLfServer.hh"

#include "utilities.hh"

#include "psdaq/trigger/Trigger.hh"
#include "psdaq/trigger/utilities.hh"
#include "psdaq/service/MetricExporter.hh"
#include "psdaq/service/Collection.hh"
#include "psdaq/service/Dl.hh"
#include "psalg/utils/SysLog.hh"
#include "xtcdata/xtc/Dgram.hh"

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
#include <exception>
#include <algorithm>                    // For std::fill()
#include <set>                          // For multiset
#include <Python.h>

#include "rapidjson/document.h"

using namespace rapidjson;
using namespace XtcData;
using namespace Pds;
using namespace Pds::Trg;

using json     = nlohmann::json;
using logging  = psalg::SysLog;
using string_t = std::string;

static const int      CORE_0          = 18; // devXXX: 11, devXX:  7, accXX:  9
static const int      CORE_1          = 19; // devXXX: 12, devXX: 19, accXX: 21
static const string_t TRIGGER_DETNAME = "tmoTeb";
static const unsigned PROM_PORT_BASE  = 9200; // Prometheus port
static const unsigned MAX_PROM_PORTS  = 100;

static struct sigaction      lIntAction;
static volatile sig_atomic_t lRunning = 1;

void sigHandler( int signal )
{
  static unsigned callCount(0);

  if (callCount == 0)
  {
    logging::info("\nShutting down");

    lRunning = 0;
  }

  if (callCount++)
  {
    logging::critical("Aborting on 2nd ^C...");
    ::abort();
  }
}


namespace Pds {
  namespace Eb {

    class Teb : public EbAppBase
    {
    public:
      Teb(const EbParams& prms, const std::shared_ptr<MetricExporter>& exporter);
    public:
      int      configure(const EbParams&, Trigger* object, unsigned prescale);
      void     run();
    public:                         // For EventBuilder
      virtual
      void     process(EbEvent* event);
    private:
      void     _tryPost(const EbDgram& dg);
      void     _post(const Batch&);
      uint64_t _receivers(const EbDgram& ctrb) const;
      void     _shutdown();
    private:
      std::vector<EbLfCltLink*>    _l3Links;
      EbLfServer                   _mrqTransport;
      std::vector<EbLfSvrLink*>    _mrqLinks;
      BatchManager                 _batMan;
      std::multiset<Batch*, Batch> _batchList;
      unsigned                     _id;
      const unsigned               _verbose;
    private:
      u64arr_t                     _rcvrs;
      //uint64_t                     _trimmed;
      Trigger*                     _trigger;
      unsigned                     _prescale;
    private:
      unsigned                     _wrtCounter;
    private:
      uint64_t                     _eventCount;
      uint64_t                     _batchCount;
      uint64_t                     _writeCount;
      uint64_t                     _monitorCount;
      uint64_t                     _prescaleCount;
    private:
      const EbParams&              _prms;
      EbLfClient                   _l3Transport;
    };
  };
};


using namespace Pds::Eb;

Teb::Teb(const EbParams& prms, const std::shared_ptr<MetricExporter>& exporter) :
  EbAppBase     (prms, BATCH_DURATION, MAX_ENTRIES, MAX_BATCHES),
  _l3Links      (),
  _mrqTransport (prms.verbose),
  _mrqLinks     (),
  _batMan       (prms.maxResultSize),
  _id           (-1),
  _verbose      (prms.verbose),
  //_trimmed      (0),
  _trigger      (nullptr),
  _eventCount   (0),
  _batchCount   (0),
  _writeCount   (0),
  _monitorCount (0),
  _prescaleCount(0),
  _prms         (prms),
  _l3Transport  (prms.verbose)
{
  std::map<std::string, std::string> labels{{"partition", std::to_string(prms.partition)}};
  exporter->add("TEB_EvtRt",  labels, MetricType::Rate,    [&](){ return _eventCount;             });
  exporter->add("TEB_EvtCt",  labels, MetricType::Counter, [&](){ return _eventCount;             });
  exporter->add("TEB_BatCt",  labels, MetricType::Counter, [&](){ return _batchCount;             }); // Outbound
  exporter->add("TEB_BtAlCt", labels, MetricType::Counter, [&](){ return _batMan.batchAllocCnt(); });
  exporter->add("TEB_BtFrCt", labels, MetricType::Counter, [&](){ return _batMan.batchFreeCnt();  });
  exporter->add("TEB_BtWtg",  labels, MetricType::Gauge,   [&](){ return _batMan.batchWaiting();  });
  exporter->add("TEB_EpAlCt", labels, MetricType::Counter, [&](){ return  epochAllocCnt();        });
  exporter->add("TEB_EpFrCt", labels, MetricType::Counter, [&](){ return  epochFreeCnt();         });
  exporter->add("TEB_EvAlCt", labels, MetricType::Counter, [&](){ return  eventAllocCnt();        });
  exporter->add("TEB_EvFrCt", labels, MetricType::Counter, [&](){ return  eventFreeCnt();         });
  exporter->add("TEB_TxPdg",  labels, MetricType::Gauge,   [&](){ return _l3Transport.pending();  });
  exporter->add("TEB_RxPdg",  labels, MetricType::Gauge,   [&](){ return  rxPending();            });
  exporter->add("TEB_BtInCt", labels, MetricType::Counter, [&](){ return  bufferCnt();            }); // Inbound
  exporter->add("TEB_FxUpCt", labels, MetricType::Counter, [&](){ return  fixupCnt();             });
  exporter->add("TEB_ToEvCt", labels, MetricType::Counter, [&](){ return  tmoEvtCnt();            });
  exporter->add("TEB_WrtCt",  labels, MetricType::Counter, [&](){ return  _writeCount;            });
  exporter->add("TEB_MonCt",  labels, MetricType::Counter, [&](){ return  _monitorCount;          });
  exporter->add("TEB_PsclCt", labels, MetricType::Counter, [&](){ return  _prescaleCount;         });
}

int Teb::configure(const EbParams& prms,
                   Trigger*        object,
                   unsigned        prescale)
{
  _id    = prms.id;
  _rcvrs = prms.receivers;

  int rc;
  if ( (rc = EbAppBase::configure(prms)) )  return rc;

  _trigger    = object;
  _prescale   = prescale - 1;           // Be zero based
  _wrtCounter = _prescale;              // Reset prescale counter

  void*  region  = _batMan.batchRegion();
  size_t regSize = _batMan.batchRegionSize();

  _l3Links.resize(prms.addrs.size());
  for (unsigned i = 0; i < _l3Links.size(); ++i)
  {
    const char*    addr = prms.addrs[i].c_str();
    const char*    port = prms.ports[i].c_str();
    EbLfCltLink*   link;
    const unsigned tmo(120000);         // Milliseconds
    if ( (rc = _l3Transport.connect(&link, addr, port, _id, tmo)) )
    {
      logging::error("%s:\n  Error connecting to Ctrb at %s:%s",
                     __PRETTY_FUNCTION__, addr, port);
      return rc;
    }
    unsigned rmtId = link->id();
    _l3Links[rmtId] = link;

    logging::debug("Outbound link with Ctrb ID %d connected", rmtId);

    if ( (rc = link->prepare(region, regSize)) )
    {
      logging::error("%s:\n  Failed to prepare link with Ctrb ID %d",
                     __PRETTY_FUNCTION__, rmtId);
      return rc;
    }

    logging::info("Outbound link with Ctrb ID %d connected and configured",
                  rmtId);
  }

  if ( (rc = _mrqTransport.initialize(prms.ifAddr, prms.mrqPort, prms.numMrqs)) )
  {
    logging::error("%s:\n  Failed to initialize MonReq EbLfServer",
                   __PRETTY_FUNCTION__);
    return rc;
  }

  _mrqLinks.resize(prms.numMrqs);
  for (unsigned i = 0; i < _mrqLinks.size(); ++i)
  {
    EbLfSvrLink*   link;
    const unsigned tmo(120000);         // Milliseconds
    if ( (rc = _mrqTransport.connect(&link, _id, tmo)) )
    {
      logging::error("%s:\n  Error connecting to a Mon Requestor",
                     __PRETTY_FUNCTION__);
      return rc;
    }
    unsigned rmtId = link->id();
    _mrqLinks[rmtId] = link;

    logging::debug("Inbound link with Mon Requestor ID %d connected", rmtId);

    if ( (rc = link->prepare()) )
    {
      logging::error("%s:\n  Failed to prepare link with Mon Requestor ID %d",
                     __PRETTY_FUNCTION__, rmtId);
      return rc;
    }
    if (link->postCompRecv())
    {
      logging::error("%s:\n  Failed to post CQ buffers for Mon Requestor ID %d",
                     __PRETTY_FUNCTION__, rmtId);
    }

    logging::info("Inbound link with Mon Requestor ID %d connected and configured",
                  rmtId);
  }

  return 0;
}

void Teb::run()
{
  pinThread(pthread_self(), _prms.core[0]);

  logging::info("TEB thread is starting");

  //_trimmed       = 0;
  _eventCount    = 0;
  _batchCount    = 0;
  _writeCount    = 0;
  _monitorCount  = 0;
  _prescaleCount = 0;

  while (lRunning)
  {
    if (EbAppBase::process() < 0)
    {
      if (checkEQ() == -FI_ENOTCONN)  break;
    }
  }

  _shutdown();

  logging::info("TEB thread is exiting");
}

void Teb::_shutdown()
{
  for (auto it = _mrqLinks.begin(); it != _mrqLinks.end(); ++it)
  {
    _mrqTransport.disconnect(*it);
  }
  _mrqLinks.clear();
  _mrqTransport.shutdown();

  for (auto it = _l3Links.begin(); it != _l3Links.end(); ++it)
  {
    _l3Transport.disconnect(*it);
  }
  _l3Links.clear();

  EbAppBase::shutdown();

  _batMan.dump();
  _batMan.shutdown();

  _id = -1;
  _rcvrs.fill(0);
}

void Teb::process(EbEvent* event)
{
  // Accumulate output datagrams (result) from the event builder into a batch
  // datagram.  When the batch completes, post it to the contributors.

  if (_verbose >= VL_DETAILED)
  {
    static unsigned cnt = 0;
    printf("Teb::process event dump:\n");
    event->dump(++cnt);
  }
  ++_eventCount;

  const EbDgram& dg = *event->creator();

  if (ImmData::rsp(ImmData::flg(event->parameter())) == ImmData::Response)
  {
    Batch*       batch = _batMan.fetch(dg.pulseId());
    ResultDgram& rdg   = *new(batch->allocate()) ResultDgram(dg, _id);

    rdg.xtc.damage.increase(event->damage().value());

    // Accumulate the list of ctrbs to this batch
    batch->accumRcvrs(_receivers(dg));
    batch->accumRogs(dg);

    if (dg.isEvent())
    {
      // Present event contributions to "user" code for building a result datagram
      _trigger->event(event->begin(), event->end(), rdg); // Consume

      unsigned line = 0;                // Revisit: For future expansion

      // Handle prescale
      if (!rdg.persist(line) && !_wrtCounter--)
      {
        _prescaleCount++;

        rdg.prescale(line, true);
        _wrtCounter = _prescale;
      }

      if (rdg.persist())  _writeCount++;
      if (rdg.monitor())
      {
        _monitorCount++;

        uint64_t data;
        int      rc = _mrqTransport.poll(&data);
        if (rc < 0)  rdg.monitor(line, false);
        else
        {
          rdg.monBufNo(data);

          rc = _mrqLinks[ImmData::src(data)]->postCompRecv();
          if (rc)
          {
            logging::error("%s:\n  Failed to post CQ buffers: %d",
                           __PRETTY_FUNCTION__, rc);
          }
        }
      }
    }

    if (_verbose >= VL_EVENT) // || rdg.monitor())
    {
      uint64_t  pid = rdg.pulseId();
      unsigned  idx = Batch::batchNum(pid);
      unsigned  ctl = rdg.control();
      size_t    sz  = sizeof(rdg) + rdg.xtc.sizeofPayload();
      unsigned  src = rdg.xtc.src.value();
      unsigned  env = rdg.env;
      uint32_t* pld = reinterpret_cast<uint32_t*>(rdg.xtc.payload());
      printf("TEB processed                result  [%5d] @ "
             "%16p, ctl %02x, pid %014lx, env %08x, sz %6zd, src %2d, res [%08x, %08x]\n",
             idx, &rdg, ctl, pid, env, sz, src, pld[0], pld[1]);
    }
  }

  _tryPost(dg);
}

void Teb::_tryPost(const EbDgram& dg)
{
  const auto pid   = dg.pulseId();
  const auto idx   = Batch::batchNum(pid);
  auto       cur   = _batMan.batch(idx);
  bool       flush = !(dg.isEvent() || (dg.service() == TransitionId::SlowUpdate));

  for (auto it = _batchList.cbegin(); it != _batchList.cend(); )
  {
    auto batch = *it;
    auto rogs  = batch->rogsRem(dg);    // Take down RoG bits

    if ((batch->expired(pid) && !rogs) || flush)
    {
      batch->terminate();
      _post(*batch);

      it = _batchList.erase(it);
    }
    else
    {
      ++it;
    }
    if (batch == cur)  return;          // Insert only once
  }

  if (flush)
  {
    cur->terminate();
   _post(*cur);
  }
  else
    _batchList.insert(cur);
}

void Teb::_post(const Batch& batch)
{
  uint32_t    idx    = batch.index();
  uint64_t    data   = ImmData::value(ImmData::Buffer, _id, idx);
  size_t      extent = batch.extent();
  unsigned    offset = idx * _batMan.maxBatchSize();
  const void* buffer = batch.buffer();
  uint64_t    destns = batch.receivers(); // & ~_trimmed;

  ++_batchCount;

  while (destns)
  {
    unsigned     dst  = __builtin_ffsl(destns) - 1;
    EbLfCltLink* link = _l3Links[dst];

    destns &= ~(1ul << dst);

    if (_verbose >= VL_BATCH)
    {
      uint64_t pid    = batch.id();
      void*    rmtAdx = (void*)link->rmtAdx(offset);
      printf("TEB posts          %9ld result  [%5d] @ "
             "%16p,         pid %014lx,               sz %6zd, dst %2d @ %16p\n",
             _batchCount, idx, buffer, pid, extent, dst, rmtAdx);
    }

    if (int rc = link->post(buffer, extent, offset, data) < 0)
    {
      if (rc != -FI_ETIMEDOUT)  break;  // Revisit: Right thing to do?

      // If we were to trim, here's how to do it.  For now, we don't.
      //static unsigned retries = 0;
      //trim(dst);
      //if (retries++ == 5)  { _trimmed |= 1ul << dst; retries = 0; }
      //printf("%s:  link->post() to %d returned %d, trimmed = %016lx\n",
      //       __PRETTY_FUNCTION__, dst, rc, _trimmed);
    }
  }

  // Revisit: The following deallocation constitutes a race with the posts to
  // the transport above as the batch's memory cannot be allowed to be reused
  // for a subsequent batch before the transmit completes.  Waiting for
  // completion here would impact performance.  Since there are many batches,
  // and only one is active at a time, a previous batch will have completed
  // transmitting before the next one starts (or the subsequent transmit won't
  // start), thus making it safe to "pre-delete" it here.
  _batMan.release(&batch);
}

uint64_t Teb::_receivers(const EbDgram& ctrb) const
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
  unsigned groups    = ctrb.readoutGroups();

  while (groups)
  {
    unsigned group = __builtin_ffs(groups) - 1;
    groups &= ~(1 << group);

    receivers |= _rcvrs[group];
  }
  return receivers;
}


class TebApp : public CollectionApp
{
public:
  TebApp(const std::string& collSrv, EbParams&);
  virtual ~TebApp();
public:                                 // For CollectionApp
  json connectionInfo() override;
  void handleConnect(const json& msg) override;
  void handleDisconnect(const json& msg) override;
  void handlePhase1(const json& msg) override;
  void handleReset(const json& msg) override;
private:
  int  _configure(const json& msg);
  int  _parseConnectionParams(const json& msg);
  void _printParams(const EbParams& prms, unsigned groups) const;
  void _printGroups(unsigned groups, const u64arr_t& array) const;
  void _buildContract(const Document& top);
private:
  EbParams&                            _prms;
  std::unique_ptr<prometheus::Exposer> _exposer;
  std::shared_ptr<MetricExporter>      _exporter;
  std::unique_ptr<Teb>                 _teb;
  std::thread                          _appThread;
  json                                 _connectMsg;
  Trg::Factory<Trg::Trigger>           _factory;
  uint16_t                             _groups;
};

TebApp::TebApp(const std::string& collSrv,
               EbParams&          prms) :
  CollectionApp(collSrv, prms.partition, "teb", prms.alias),
  _prms        (prms)
{
  Py_Initialize();

  logging::info("Ready for transitions");
}

TebApp::~TebApp()
{
  Py_Finalize();
}

json TebApp::connectionInfo()
{
  // Allow the default NIC choice to be overridden
  std::string ip = _prms.ifAddr.empty() ? getNicIp() : _prms.ifAddr;
  json body = {{"connect_info", {{"nic_ip", ip}}}};
  return body;
}

void TebApp::handleConnect(const json& msg)
{
  json body = json({});
  int  rc   = _parseConnectionParams(msg["body"]);
  if (rc)
  {
    std::string errorMsg = "Error parsing connect parameters";
    body["err_info"] = errorMsg;
    logging::error("%s:\n  %s", __PRETTY_FUNCTION__, errorMsg.c_str());
  }

  // Save a copy of the json so we can use it to connect to
  // the config database on configure
  _connectMsg = msg;

  // Reply to collection with transition status
  reply(createMsg("connect", msg["header"]["msg_id"], getId(), body));
}

void TebApp::_buildContract(const Document& top)
{
  const json& body = _connectMsg["body"];

  _prms.contractors.fill(0);

  for (auto it : body["drp"].items())
  {
    unsigned    drpId   = it.value()["drp_id"];
    std::string alias   = it.value()["proc_info"]["alias"];
    size_t      found   = alias.rfind('_');
    std::string detName = alias.substr(0, found);

    if (top.HasMember(detName.c_str()))
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
  std::string&      detName = _prms.trgDetName;
  if (_prms.trgDetName.empty())  _prms.trgDetName = TRIGGER_DETNAME;

  logging::info("Fetching trigger info from ConfigDb/%s/%s\n\n",
         configAlias.c_str(), detName.c_str());

  if (Pds::Trg::fetchDocument(_connectMsg.dump(), configAlias, detName, top))
  {
    logging::error("%s:\n  Document '%s' not found in ConfigDb\n",
            __PRETTY_FUNCTION__, detName.c_str());
    return -1;
  }

  if (detName != TRIGGER_DETNAME)  _buildContract(top);

  const std::string symbol("create_consumer");
  Trigger* trigger = _factory.create(top, detName, symbol);
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
                        __PRETTY_FUNCTION__, key, detName.c_str());     \
         rc = -1; }

  unsigned prescale;  _FETCH("prescale", prescale);

# undef _FETCH

  if (_exposer)  _exposer.reset();
  unsigned port = 0;
  for (unsigned i = 0; i < MAX_PROM_PORTS; ++i) {
    try {
      port = PROM_PORT_BASE + i;
      _exposer = std::make_unique<prometheus::Exposer>("0.0.0.0:"+std::to_string(port), "/metrics", 1);
      if (i > 0) {
        if ((i < MAX_PROM_PORTS) && !_prms.prometheusDir.empty()) {
          char hostname[HOST_NAME_MAX];
          gethostname(hostname, HOST_NAME_MAX);
          std::string fileName = _prms.prometheusDir + "/drpmon_" + std::string(hostname) + "_" + std::to_string(i) + ".yaml";
          FILE* file = fopen(fileName.c_str(), "w");
          if (file) {
            fprintf(file, "- targets:\n    - '%s:%d'\n", hostname, port);
            fclose(file);
          }
          else {
            // %m will be replaced by the string strerror(errno)
            logging::error("Error creating file %s: %m", fileName.c_str());
          }
        }
        else {
          logging::warning("Could not start run-time monitoring server");
        }
      }
      break;
    }
    catch(const std::runtime_error& e) {
      logging::debug("Could not start run-time monitoring server on port %d", port);
      logging::debug("%s", e.what());
    }
  }

  if (_exporter)  _exporter.reset();
  _exporter = std::make_shared<MetricExporter>();
  if (_exposer) {
    logging::info("Providing run-time monitoring data on port %d", port);
    _exposer->RegisterCollectable(_exporter);
  }

  if (_teb)  _teb.reset();
  _teb = std::make_unique<Teb>(_prms, _exporter);
  rc = _teb->configure(_prms, trigger, prescale);

  return rc;
}

void TebApp::handlePhase1(const json& msg)
{
  json        body = json({});
  std::string key  = msg["header"]["key"];

  if (key == "configure")
  {
    // Shut down the previously running instance, if any
    if (_appThread.joinable())
    {
      lRunning = 0;

      _appThread.join();
    }

    int rc = _configure(msg);
    if (rc)
    {
      std::string errorMsg = "Phase 1 error: ";
      errorMsg += "Failed to configure";
      body["err_info"] = errorMsg;
      logging::error("%s:\n  %s", __PRETTY_FUNCTION__, errorMsg.c_str());
    }
    else
    {
      _printParams(_prms, _groups);

      lRunning = 1;

      _appThread = std::thread(&Teb::run, std::ref(*_teb));
    }
  }

  // Reply to collection with transition status
  reply(createMsg(key, msg["header"]["msg_id"], getId(), body));
}

void TebApp::handleDisconnect(const json& msg)
{
  lRunning = 0;

  if (_appThread.joinable())  _appThread.join();

  // Reply to collection with transition status
  json body = json({});
  reply(createMsg("disconnect", msg["header"]["msg_id"], getId(), body));
}

void TebApp::handleReset(const json& msg)
{
  lRunning = 0;

  if (_appThread.joinable())  _appThread.join();
}

int TebApp::_parseConnectionParams(const json& body)
{
  const unsigned numPorts    = MAX_DRPS + MAX_TEBS + MAX_TEBS + MAX_MEBS;
  const unsigned tebPortBase = TEB_PORT_BASE + numPorts * _prms.partition;
  const unsigned drpPortBase = DRP_PORT_BASE + numPorts * _prms.partition;
  const unsigned mrqPortBase = MRQ_PORT_BASE + numPorts * _prms.partition;

  printf("  TEB port range: %d - %d\n", tebPortBase, tebPortBase + MAX_TEBS - 1);
  printf("  DRP port range: %d - %d\n", drpPortBase, drpPortBase + MAX_DRPS - 1);
  printf("  MRQ port range: %d - %d\n", mrqPortBase, mrqPortBase + MAX_MEBS - 1);
  printf("\n");

  std::string id = std::to_string(getId());
  _prms.id = body["teb"][id]["teb_id"];
  if (_prms.id >= MAX_TEBS)
  {
    logging::error("TEB ID %d is out of range 0 - %d\n", _prms.id, MAX_TEBS - 1);
    return 1;
  }

  _prms.ifAddr  = body["teb"][id]["connect_info"]["nic_ip"];
  _prms.ebPort  = std::to_string(tebPortBase + _prms.id);
  _prms.mrqPort = std::to_string(mrqPortBase + _prms.id);

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
    std::string address = it.value()["connect_info"]["nic_ip"];
    if (drpId > MAX_DRPS - 1)
    {
      logging::error("DRP ID %d is out of range 0 - %d", drpId, MAX_DRPS - 1);
      return 1;
    }
    _prms.contributors |= 1ul << drpId;
    _prms.addrs.push_back(address);
    _prms.ports.push_back(std::string(std::to_string(drpPortBase + drpId)));

    auto group = unsigned(it.value()["det_info"]["readout"]);
    if (group > NUM_READOUT_GROUPS - 1)
    {
      logging::error("Readout group %d is out of range 0 - %d", group, NUM_READOUT_GROUPS - 1);
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
  }

  return 0;
}

void TebApp::_printGroups(unsigned groups, const u64arr_t& array) const
{
  while (groups)
  {
    unsigned group = __builtin_ffs(groups) - 1;
    groups &= ~(1 << group);

    printf("%d: 0x%016lx  ", group, array[group]);
  }
  printf("\n");
}

void TebApp::_printParams(const EbParams& prms, unsigned groups) const
{
  printf("Parameters of TEB ID %d:\n",                         prms.id);
  printf("  Thread core numbers:        %d, %d\n",             prms.core[0], prms.core[1]);
  printf("  Partition:                  %d\n",                 prms.partition);
  printf("  Bit list of contributors:   0x%016lx, cnt: %zd\n", prms.contributors,
                                                                std::bitset<64>(prms.contributors).count());
  printf("  Readout group contractors:  ");                    _printGroups(groups, prms.contractors);
  printf("  Readout group receivers:    ");                    _printGroups(groups, prms.receivers);
  printf("  ConfigDb trigger detName:     %s\n",                 prms.trgDetName.c_str());
  printf("  Number of MEB requestors:     %d\n",                 prms.numMrqs);
  printf("  Batch duration:               0x%014lx = %ld uS\n",  BATCH_DURATION, BATCH_DURATION);
  printf("  Batch pool depth:             %d\n",                 MAX_BATCHES);
  printf("  Max # of entries / batch:     %d\n",                 MAX_ENTRIES);
  printf("  # of contrib. buffers:        %d\n",                 MAX_LATENCY);
  printf("  Max result     EbDgram size:  %zd\n",                prms.maxResultSize);
  printf("  Max transition EbDgram size:  %zd\n",                prms.maxTrSize[0]);
  printf("\n");
}


static void usage(char *name, char *desc, const EbParams& prms)
{
  fprintf(stderr, "Usage:\n");
  fprintf(stderr, "  %s [OPTIONS]\n", name);

  if (desc)
    fprintf(stderr, "\n%s\n", desc);

  fprintf(stderr, "\nOptions:\n");

  fprintf(stderr, " %-23s %s (default: %s)\n",        "-A <interface_addr>",
          "IP address of the interface to use",       "libfabric's 'best' choice");

  fprintf(stderr, " %-23s %s (required)\n",           "-C <address>",
          "Collection server");
  fprintf(stderr, " %-23s %s (required)\n",           "-p <partition number>",
          "Partition number");
  fprintf(stderr, " %-23s %s\n",                      "-P <instrument>",
          "Instrument name");
  fprintf(stderr, " %-23s %s (default: '%s')\n"
                  " %-23s %s\n",                      "-T[<trigger 'detName'>]",
          "ConfigDb detName for trigger",             TRIGGER_DETNAME.c_str(),
          " ", "(-T without arg gives system default; n.b. no space between -T and arg)");
  fprintf(stderr, " %-23s %s (required)\n",           "-u <alias>",
          "Alias for teb process");
  fprintf(stderr, " %-23s %s\n",                      "-M <directory>",
          "Prometheus config file directory");
  fprintf(stderr, " %-23s %s (default: %d)\n",        "-1 <core>",
          "Core number for pinning App thread to",    CORE_0);
  fprintf(stderr, " %-23s %s (default: %d)\n",        "-2 <core>",
          "Core number for pinning other threads to", CORE_1);

  fprintf(stderr, " %-23s %s\n", "-v", "enable debugging output (repeat for increased detail)");
  fprintf(stderr, " %-23s %s\n", "-h", "display this help output");
}


int main(int argc, char **argv)
{
  const unsigned NO_PARTITION = unsigned(-1u);
  int            op           = 0;
  char*          instrument   = NULL;
  std::string    collSrv;
  EbParams       prms;

  prms.partition = NO_PARTITION;
  prms.core[0]   = CORE_0;
  prms.core[1]   = CORE_1;
  prms.verbose   = 0;

  while ((op = getopt(argc, argv, "C:p:P:T::A:1:2:u:M:h?v")) != -1)
  {
    switch (op)
    {
      case 'C':  collSrv            = optarg;                       break;
      case 'p':  prms.partition     = std::stoi(optarg);            break;
      case 'P':  instrument         = optarg;                       break;
      case 'T':  prms.trgDetName    = optarg ? optarg : "trigger";  break;
      case 'A':  prms.ifAddr        = optarg;                       break;
      case '1':  prms.core[0]       = atoi(optarg);                 break;
      case '2':  prms.core[1]       = atoi(optarg);                 break;
      case 'u':  prms.alias         = optarg;                       break;
      case 'M':  prms.prometheusDir = optarg;                       break;
      case 'v':  ++prms.verbose;                                    break;
      case '?':
      case 'h':
      default:
        usage(argv[0], (char*)"Trigger Event Builder application", prms);
        return 1;
    }
  }

  logging::init(instrument, prms.verbose ? LOG_DEBUG : LOG_INFO);
  logging::info("logging configured");

  if (!instrument)
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

  TebApp app(collSrv, prms);

  try
  {
    app.run();
  }
  catch (std::exception& e)
  {
    logging::critical("Application exception:\n  %s", e.what());
  }

  app.handleReset(json({}));

  return 0;
}
