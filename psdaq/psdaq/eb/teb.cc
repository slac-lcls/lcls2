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
#include "psdaq/service/SysLog.hh"
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
#include <algorithm>                    // For std::fill()
#include <set>                          // For multiset
#include <Python.h>

#include "rapidjson/document.h"

using namespace rapidjson;
using namespace XtcData;
using namespace Pds;
using namespace Pds::Trg;

using json     = nlohmann::json;
using logging  = Pds::SysLog;
using string_t = std::string;

static const int      CORE_0           = 18; // devXXX: 11, devXX:  7, accXX:  9
static const int      CORE_1           = 19; // devXXX: 12, devXX: 19, accXX: 21
static const size_t   HEADER_SIZE      = sizeof(Dgram);
static const size_t   INPUT_EXTENT     = 2; // Revisit: Number of "L3" input  data words
static const size_t   RESULT_EXTENT    = 2; // Revisit: Number of "L3" result data words
static const size_t   MAX_CONTRIB_SIZE = HEADER_SIZE + INPUT_EXTENT  * sizeof(uint32_t);
static const size_t   MAX_RESULT_SIZE  = HEADER_SIZE + RESULT_EXTENT * sizeof(uint32_t);
static const string_t TRIGGER_DETNAME  = "tmoTeb";

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

    class Teb : public EbAppBase
    {
    public:
      Teb(const EbParams& prms, std::shared_ptr<MetricExporter>& exporter);
    public:
      int      connect(const EbParams&);
      int      configure(const EbParams&, Trigger* object, unsigned prescale);
      Trigger* trigger() const { return _trigger; }
      void     run();
    public:                         // For EventBuilder
      virtual
      void     process(EbEvent* event);
    private:
      void     _tryPost(const Dgram& dg);
      void     _post(const Batch&);
      uint64_t _receivers(const Dgram& ctrb) const;
    private:
      std::vector<EbLfLink*>       _l3Links;
      EbLfServer                   _mrqTransport;
      std::vector<EbLfLink*>       _mrqLinks;
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

Teb::Teb(const EbParams& prms, std::shared_ptr<MetricExporter>& exporter) :
  EbAppBase     (prms, BATCH_DURATION, MAX_ENTRIES, MAX_BATCHES),
  _l3Links      (),
  _mrqTransport (prms.verbose),
  _mrqLinks     (),
  _batMan       (Trigger::size()), // Revisit: prms.maxResultSize),
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

int Teb::connect(const EbParams& prms)
{
  int rc;
  if ( (rc = EbAppBase::connect(prms)) )
    return rc;

  _id = prms.id;
  _l3Links.resize(prms.addrs.size());
  _rcvrs = prms.receivers;

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

int Teb::configure(const EbParams& prms,
                   Trigger*        object,
                   unsigned        prescale)
{
  EbAppBase::configure(prms);

  _trigger    = object;
  _prescale   = prescale - 1;           // Be zero based
  _wrtCounter = _prescale;              // Reset prescale counter

  return 0;
}

void Teb::run()
{
  pinThread(pthread_self(), _prms.core[0]);

  //_trimmed       = 0;
  _eventCount    = 0;
  _batchCount    = 0;
  _writeCount    = 0;
  _monitorCount  = 0;
  _prescaleCount = 0;

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
  _rcvrs.fill(0);
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

  const Dgram& dg = *event->creator();

  if (ImmData::rsp(ImmData::flg(event->parameter())) == ImmData::Response)
  {
    Batch*       batch = _batMan.fetch(dg);
    ResultDgram& rdg   = *new(batch->allocate()) ResultDgram(dg, dg.xtc.src.value());

    rdg.xtc.damage.increase(event->damage().value());

    // Accumulate the list of ctrbs to this batch
    batch->accumRcvrs(_receivers(dg));
    batch->accumRogs(dg);

    if (dg.seq.isEvent())
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
            fprintf(stderr, "%s:\n  Failed to post CQ buffers: %d\n",
                    __PRETTY_FUNCTION__, rc);
          }
        }
      }
    }

    if (_verbose > 2) // || rdg.monitor())
    {
      uint64_t  pid = rdg.seq.pulseId().value();
      unsigned  idx = Batch::batchNum(pid);
      unsigned  ctl = rdg.seq.pulseId().control();
      size_t    sz  = sizeof(rdg) + rdg.xtc.sizeofPayload();
      unsigned  src = rdg.xtc.src.value();
      unsigned  env = rdg.env;
      printf("TEB processed                result  [%5d] @ "
             "%16p, ctl %02x, pid %014lx, sz %6zd, src %2d, env %08x, res [%08x, %08x]\n",
             idx, &rdg, ctl, pid, sz, src, env, rdg.persist(), rdg.monitor());
    }
  }

  _tryPost(dg);
}

void Teb::_tryPost(const Dgram& dg)
{
  const auto pid   = dg.seq.pulseId().value();
  const auto idx   = Batch::batchNum(pid);
  auto       cur   = _batMan.batch(idx);
  bool       flush = !(dg.seq.isEvent() || (dg.seq.service() == TransitionId::SlowUpdate));

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
    unsigned  dst  = __builtin_ffsl(destns) - 1;
    EbLfLink* link = _l3Links[dst];

    destns &= ~(1ul << dst);

    if (_verbose)
    {
      uint64_t pid    = batch.id();
      void*    rmtAdx = (void*)link->rmtAdx(offset);
      printf("TEB posts          %9ld result  [%5d] @ "
             "%16p,         pid %014lx, sz %6zd, dst %2d @ %16p\n",
             _batchCount, idx, buffer, pid, extent, dst, rmtAdx);
    }

    int rc;
    if ( (rc = link->post(buffer, extent, offset, data)) < 0)
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

uint64_t Teb::_receivers(const Dgram& ctrb) const
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

static void _printParams(EbParams& prms, unsigned groups)
{
  printf("Parameters of TEB ID %d:\n",                        prms.id);
  printf("  Thread core numbers:         %d, %d\n",           prms.core[0], prms.core[1]);
  printf("  Partition:                   %d\n",               prms.partition);
  printf("  Bit list of contributors:  0x%016lx, cnt: %zd\n", prms.contributors,
                                                              std::bitset<64>(prms.contributors).count());
  printf("  Readout group contractors:   ");                  _printGroups(groups, prms.contractors);
  printf("  Readout group receivers:     ");                  _printGroups(groups, prms.receivers);
  printf("  ConfigDb trigger detName:    %s\n",               prms.trgDetName.c_str());
  printf("  Number of MEB requestors:    %d\n",               prms.numMrqs);
  printf("  Batch duration:            0x%014lx = %ld uS\n",  BATCH_DURATION, BATCH_DURATION);
  printf("  Batch pool depth:            %d\n",               MAX_BATCHES);
  printf("  Max # of entries / batch:    %d\n",               MAX_ENTRIES);
  printf("  # of contrib. buffers:       %d\n",               MAX_LATENCY);
  printf("  Max result     Dgram size:   %zd\n",              prms.maxResultSize);
  printf("  Max transition Dgram size:   %zd\n",              prms.maxTrSize[0]);
  printf("\n");
}

class TebApp : public CollectionApp
{
public:
  TebApp(const std::string& collSrv, EbParams&, std::shared_ptr<MetricExporter>&);
  virtual ~TebApp();
public:                                 // For CollectionApp
  json connectionInfo() override;
  void handleConnect(const json& msg) override;
  void handleDisconnect(const json& msg) override;
  void handlePhase1(const json& msg) override;
  void handleReset(const json& msg) override;
private:
  int  _connect(const json& msg);
  int  _configure(const json& msg);
  int  _parseConnectionParams(const json& msg);
  void _buildContract(const Document& top);
private:
  EbParams&                  _prms;
  Teb                        _teb;
  std::thread                _appThread;
  json                       _connectMsg;
  Trg::Factory<Trg::Trigger> _factory;
  uint16_t                   _groups;
};

TebApp::TebApp(const std::string&               collSrv,
               EbParams&                        prms,
               std::shared_ptr<MetricExporter>& exporter) :
  CollectionApp(collSrv, prms.partition, "teb", prms.alias),
  _prms        (prms),
  _teb         (prms, exporter)
{
  Py_Initialize();
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

int TebApp::_connect(const json& msg)
{
  int rc = _parseConnectionParams(msg["body"]);
  if (rc)  return rc;

  rc = _teb.connect(_prms);
  if (rc)  return rc;

  lRunning = 1;

  _appThread = std::thread(&Teb::run, std::ref(_teb));

  return 0;
}

void TebApp::handleConnect(const json& msg)
{
  json body = json({});
  int  rc   = _connect(msg);
  if (rc)
  {
    std::string errorMsg = "Failed to connect";
    body["err_info"] = errorMsg;
    fprintf(stderr, "%s:\n  %s\n", __PRETTY_FUNCTION__, errorMsg.c_str());
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
    fprintf(stderr, "%s:\n  Failed to create Trigger\n",
            __PRETTY_FUNCTION__);
    return -1;
  }

  if (trigger->configure(_connectMsg, top))
  {
    fprintf(stderr, "%s:\n  Failed to configure Trigger\n",
            __PRETTY_FUNCTION__);
    return -1;
  }

# define _FETCH(key, item)                                               \
  if (top.HasMember(key))  item = top[key].GetUint();                    \
  else { fprintf(stderr, "%s:\n  Key '%s' not found in Document %s\n",   \
                 __PRETTY_FUNCTION__, key, detName.c_str());  rc = -1; }

  unsigned prescale;  _FETCH("prescale", prescale);

# undef _FETCH

  _teb.configure(_prms, trigger, prescale);

  return rc;
}

void TebApp::handlePhase1(const json& msg)
{
  json        body = json({});
  std::string key  = msg["header"]["key"];

  if (key == "configure")
  {
    int rc = _configure(msg);
    if (rc)
    {
      std::string errorMsg = "Phase 1 error: ";
      errorMsg += "Failed to configure TEB";
      body["err_info"] = errorMsg;
      logging::error("%s:\n  %s", __PRETTY_FUNCTION__, errorMsg.c_str());
    }

    _printParams(_prms, _groups);
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
  std::fill(vec.begin(), vec.end(), MAX_CONTRIB_SIZE); // Same for all contributors

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
  EbParams       prms { /* .ifAddr        = */ { }, // Network interface to use
                        /* .ebPort        = */ { },
                        /* .mrqPort       = */ { },
                        /* .partition     = */ NO_PARTITION,
                        /* .alias         = */ { }, // Unique name passed on cmd line
                        /* .id            = */ -1u,
                        /* .contributors  = */ 0,   // DRPs
                        /* .contractors   = */ { },
                        /* .receivers     = */ { },
                        /* .addrs         = */ { }, // Result dst addr served by Ctrbs
                        /* .ports         = */ { }, // Result dst port served by Ctrbs
                        /* .maxTrSize     = */ { }, // Filled in at connect
                        /* .maxResultSize = */ MAX_RESULT_SIZE,
                        /* .numMrqs       = */ 0,   // Number of Mon requestors
                        /* .trgDetName    = */ { },
                        /* .core          = */ { CORE_0, CORE_1 },
                        /* .verbose       = */ 0 };

  while ((op = getopt(argc, argv, "C:p:P:T::A:1:2:u:h?v")) != -1)
  {
    switch (op)
    {
      case 'C':  collSrv         = optarg;                       break;
      case 'p':  prms.partition  = std::stoi(optarg);            break;
      case 'P':  instrument      = optarg;                       break;
      case 'T':  prms.trgDetName = optarg ? optarg : "trigger";  break;
      case 'A':  prms.ifAddr     = optarg;                       break;
      case '1':  prms.core[0]    = atoi(optarg);                 break;
      case '2':  prms.core[1]    = atoi(optarg);                 break;
      case 'u':  prms.alias      = optarg;                       break;
      case 'v':  ++prms.verbose;                                 break;
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
    fprintf(stderr, "Failed to set up ^C handler\n");

  // Event builder sorts contributions into a time ordered list
  // Then calls the user's process() with complete events to build the result datagram
  // Post completed result batches to the outlet

  // Iterate over contributions in the batch
  // Event build them according to their trigger group

  std::unique_ptr<prometheus::Exposer> exposer;
  try
  {
    exposer = std::make_unique<prometheus::Exposer>("0.0.0.0:9200", "/metrics", 1);
  }
  catch(const std::runtime_error& e)
  {
    logging::warning("Error opening monitoring port.  Monitoring disabled.");
    std::cout<<e.what()<<std::endl;
  }

  auto exporter = std::make_shared<MetricExporter>();

  TebApp app(collSrv, prms, exporter);

  if (exposer)
  {
    exposer->RegisterCollectable(exporter);
  }

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
