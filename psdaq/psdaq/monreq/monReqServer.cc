#include "psalg/shmem/XtcMonitorServer.hh"

#include "psdaq/eb/eb.hh"
#include "psdaq/eb/EbAppBase.hh"
#include "psdaq/eb/EbEvent.hh"

#include "psdaq/eb/EbLfClient.hh"

#include "psdaq/eb/utilities.hh"

#include "psdaq/service/kwargs.hh"
#include "psdaq/service/Fifo.hh"
#include "psdaq/service/GenericPool.hh"
#include "psdaq/service/Collection.hh"
#include "psdaq/service/MetricExporter.hh"
#include "psalg/utils/SysLog.hh"

#include <signal.h>
#include <errno.h>
#include <unistd.h>                     // For getopt(), gethostname()
#include <string.h>
#include <vector>
#include <bitset>
#include <iostream>
#include <sstream>
#include <atomic>
#include <climits>                      // For HOST_NAME_MAX

static const int      CORE_0               = -1; // devXXX: 18, devXX:  7, accXX:  9
static const int      CORE_1               = -1; // devXXX: 19, devXX: 19, accXX: 21
static const unsigned EPOCH_DURATION       = 8;  // 1 per xferBuffer
static const unsigned NUMBEROF_XFERBUFFERS = 8;  // Value corresponds to ctrb:maxEvents

using namespace XtcData;
using namespace psalg::shmem;
using namespace Pds;
using namespace Pds::Eb;

using json     = nlohmann::json;
using logging  = psalg::SysLog;
using u64arr_t = std::array<uint64_t, NUM_READOUT_GROUPS>;

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
    struct MebParams : public EbParams
    {
      std::string tag;
      unsigned    nevqueues;
      bool        ldist;
      unsigned    maxBufferSize;        // Maximum built event size
      unsigned    numEvBuffers;         // Number of event buffers
    };
  };


  class MyXtcMonitorServer : public XtcMonitorServer {
  public:
    MyXtcMonitorServer(std::vector<EbLfCltLink*>& links,
                       uint64_t&                  requestCount,
                       const MebParams&           prms) :
      XtcMonitorServer(prms.tag.c_str(),
                       prms.maxBufferSize,
                       prms.numEvBuffers,
                       prms.nevqueues),
      _sizeofBuffers(prms.maxBufferSize),
      _iTeb         (0),
      _mrqLinks     (links),
      _requestCount (requestCount),
      _bufFreeList  (prms.numEvBuffers),
      _prms         (prms)
    {
      for (unsigned i = 0; i < _bufFreeList.size(); ++i)
      {
        if (_bufFreeList.push(i))
        {
          logging::critical("%s:\n  _bufFreeList.push(%u) failed", __PRETTY_FUNCTION__, i);
          throw "_bufFreeList.push() failed";
        }
      }

      _init();
    }
    virtual ~MyXtcMonitorServer()
    {
    }
    const size_t& bufListCount() const
    {
      return _bufFreeList.count();
    }
  private:
    virtual void _copyDatagram(Dgram* dg, char* buf)
    {
      if (unlikely(_prms.verbose >= VL_EVENT))
        printf("_copyDatagram:   dg %p, ts %u.%09u to %p\n",
               dg, dg->time.seconds(), dg->time.nanoseconds(), buf);

      // The dg payload is a directory of contributions to the built event.
      // Iterate over the directory and construct, in shared memory, the event
      // datagram (odg) from the contribution XTCs
      const EbDgram** const  last = (const EbDgram**)dg->xtc.next();
      const EbDgram*  const* ctrb = (const EbDgram**)dg->xtc.payload();
      Dgram*                 odg  = new((void*)buf) Dgram(**ctrb); // Not an EbDgram!
      odg->xtc.src      = XtcData::Src(XtcData::Level::Event);
      odg->xtc.contains = XtcData::TypeId(XtcData::TypeId::Parent, 0);
      do
      {
        const EbDgram* idg = *ctrb;

        odg->xtc.damage.increase(idg->xtc.damage.value());

        size_t   oSz  = sizeof(*odg) + odg->xtc.sizeofPayload();
        uint32_t iExt = idg->xtc.extent;
        if (oSz + iExt > _sizeofBuffers)
        {
          logging::debug("Truncated: Buffer of size %zu is too small to add Xtc of size %zu\n",
                         _sizeofBuffers, iExt);
          odg->xtc.damage.increase(XtcData::Damage::Truncated);
          iExt = _sizeofBuffers - oSz;
        }
        buf = (char*)odg->xtc.alloc(iExt);
        memcpy(buf, &idg->xtc, iExt);
      }
      while (++ctrb != last);
    }

    virtual void _deleteDatagram(Dgram* dg) // Not called for transitions
    {
      unsigned idx = (dg->env >> 16) & 0xff;

      if (unlikely(_prms.verbose >= VL_EVENT))
        printf("_deleteDatagram: dg %p, ts %u.%09u, idx %u\n",
               dg, dg->time.seconds(), dg->time.nanoseconds(), idx);

      if (idx >= _bufFreeList.size())
      {
        logging::warning("deleteDatagram: Unexpected index %08x", idx);
      }

      for (unsigned i = 0; i < _bufFreeList.count(); ++i)
      {
        if (idx == _bufFreeList.peek(i))
        {
          logging::error("Attempted double free of list entry %u: idx %u, dg %p, ts %u.%09u, svc %s",
                         i, idx, dg, dg->time.seconds(), dg->time.nanoseconds(), TransitionId::name(dg->service()));
          // Does the dg still need to be freed?  Apparently so.
          Pool::free((void*)dg);
          return;
        }
      }
      if (_bufFreeList.push(idx))
      {
        logging::error("_bufFreeList.push(%u) failed, count %zd", idx, _bufFreeList.count());
        for (unsigned i = 0; i < _bufFreeList.size(); ++i)
        {
          printf("Free list entry %u: %u\n", i, _bufFreeList.peek(i));
        }
      }

      Pool::free((void*)dg);
    }

    virtual void _requestDatagram()
    {
      //printf("_requestDatagram\n");

      unsigned data;
      if (_bufFreeList.pop(data))
      {
        logging::error("%s:\n  No free buffers available", __PRETTY_FUNCTION__);
        return;
      }

      //printf("_requestDatagram: _bufFreeList.pop(): %u, count = %zd\n", data, _bufFreeList.count());

      data = ImmData::value(ImmData::Buffer, _prms.id, data);

      int rc = -1;
      for (unsigned i = 0; i < _mrqLinks.size(); ++i)
      {
        // Round robin through Trigger Event Builders
        unsigned iTeb = _iTeb++;
        if (_iTeb == _mrqLinks.size())  _iTeb = 0;

        rc = _mrqLinks[iTeb]->EbLfLink::post(nullptr, 0, data);

        if (unlikely(_prms.verbose >= VL_EVENT))
          printf("_requestDatagram: Post %u EB[iTeb %u], value %08x, rc %d\n",
                 i, iTeb, data, rc);

        if (rc == 0)
        {
          ++_requestCount;
          break;            // Break if message was delivered
        }
      }
      if (rc)
      {
        logging::error("%s:\n  Unable to post request to any TEB: rc %d, data %u = %08x",
                       __PRETTY_FUNCTION__, rc, data, data);
        // Revisit: Is this fatal or ignorable?
      }
    }

  private:
    unsigned                   _sizeofBuffers;
    unsigned                   _iTeb;
    std::vector<EbLfCltLink*>& _mrqLinks;
    uint64_t&                  _requestCount;
    FifoMT<unsigned>           _bufFreeList;
    const MebParams&           _prms;
  };


  class Meb : public EbAppBase
  {
  public:
    Meb(const MebParams& prms, const MetricExporter_t& exporter);
    virtual ~Meb();
  public:
    int  resetCounters();
    int  connect();
    int  configure();
    int  beginrun();
    void unconfigure();
    void disconnect();
    void run();
  private:                              // For EventBuilder
    virtual
    void process(EbEvent* event);
  private:
    std::unique_ptr<MyXtcMonitorServer> _apps;
    std::vector<EbLfCltLink*>           _mrqLinks;
    std::unique_ptr<GenericPool>        _pool;
    uint64_t                            _eventCount;
    uint64_t                            _requestCount;
    const MebParams&                    _prms;
    EbLfClient                          _mrqTransport;
  };
};


Meb::Meb(const MebParams&        prms,
         const MetricExporter_t& exporter) :
  EbAppBase    (prms, exporter, "MEB", EPOCH_DURATION, 1, prms.numEvBuffers),
  _eventCount  (0),
  _requestCount(0),
  _prms        (prms),
  _mrqTransport(prms.verbose, prms.kwargs)
{
  std::map<std::string, std::string> labels{{"instrument", prms.instrument},
                                            {"partition", std::to_string(prms.partition)},
                                            {"detname", prms.alias},
                                            {"alias", prms.alias}};
  exporter->add("MEB_EvtRt", labels, MetricType::Rate,    [&](){ return _eventCount;      });
  exporter->add("MEB_EvtCt", labels, MetricType::Counter, [&](){ return _eventCount;      });
  exporter->add("MEB_ReqRt", labels, MetricType::Rate,    [&](){ return _requestCount;    });
  exporter->add("MEB_ReqCt", labels, MetricType::Counter, [&](){ return _requestCount;    });
  exporter->add("MRQ_BufCt", labels, MetricType::Gauge,   [&](){ return _apps ? _apps->bufListCount() : 0; });
  exporter->constant("MRQ_BufCtMax", labels, prms.numEvBuffers);
}

Meb::~Meb()
{
}

int Meb::resetCounters()
{
  EbAppBase::resetCounters();

  _eventCount   = 0;
  _requestCount = 0;

  return 0;
}

void Meb::disconnect()
{
  for (auto link : _mrqLinks)  _mrqTransport.disconnect(link);
  _mrqLinks.clear();

  EbAppBase::disconnect();
}

void Meb::unconfigure()
{
  if (_pool)
  {
    printf("Directory datagram pool:\n");
    _pool->dump();
  }
  _pool.reset();

  _apps.reset();

  EbAppBase::unconfigure();
}

int Meb::connect()
{
  _mrqLinks.resize(_prms.addrs.size());

  // Make a guess at the size of the Input entries
  size_t inpSizeGuess = 128*1024;

  int rc = EbAppBase::connect(_prms, inpSizeGuess);
  if (rc)  return rc;

  rc = linksConnect(_mrqTransport, _mrqLinks, _prms.addrs, _prms.ports, "TEB");
  if (rc)  return rc;

  return 0;
}

int Meb::configure()
{
  // Create pool for transferring events to MyXtcMonitorServer
  unsigned entries = std::bitset<64>(_prms.contributors).count();
  size_t   size    = sizeof(Dgram) + entries * sizeof(Dgram*);
  _pool = std::make_unique<GenericPool>(size, 1 + _prms.numEvBuffers); // +1 for Transitions

  int rc = EbAppBase::configure(_prms);
  if (rc)  return rc;

  rc = linksConfigure(_mrqLinks, _prms.id, "TEB");
  if (rc)  return rc;

  _apps = std::make_unique<MyXtcMonitorServer>(_mrqLinks, _requestCount, _prms);

  _apps->distribute(_prms.ldist);

  return 0;
}

int Meb::beginrun()
{
  _eventCount = 0;

  return 0;
}

void Meb::run()
{
  logging::info("MEB thread started");

  int rc = pinThread(pthread_self(), _prms.core[0]);
  if (rc != 0)
  {
    logging::error("%s:\n  Error from pinThread:\n  %s",
                   __PRETTY_FUNCTION__, strerror(rc));
  }

  while (lRunning)
  {
    if (EbAppBase::process() < 0)
    {
      if (checkEQ() == -FI_ENOTCONN)
      {
        logging::critical("MEB thread lost connection with DRP(s)");
        break;
      }
    }
  }

  logging::info("MEB thread finished");
}

void Meb::process(EbEvent* event)
{
  if (_prms.verbose >= VL_DETAILED)
  {
    static unsigned cnt = 0;
    printf("Meb::process event dump:\n");
    event->dump(++cnt);
  }
  ++_eventCount;

  // Create a Dgram with a payload that is a directory of contribution
  // Dgrams to the built event in order to avoid first assembling the
  // datagram from its contributions in a temporary buffer, only to then
  // copy it into shmem in _copyDatagram() above.  Since the contributions,
  // and thus the full datagram, can be quite large, this would amount to
  // a lot of copying
  size_t   sz     = (event->end() - event->begin()) * sizeof(*(event->begin()));
  unsigned idx    = ImmData::idx(event->parameter());
  void*    buffer = _pool->alloc(sizeof(Dgram) + sz);
  if (!buffer)
  {
    logging::critical("%s:\n  Dgram pool allocation of size %zd failed:",
                      __PRETTY_FUNCTION__, sizeof(Dgram) + sz);
    printf("Directory datagram pool\n");
    _pool->dump();
    printf("Meb::process event dump:\n");
    event->dump(-1);
    throw "Dgram pool exhausted";
  }

  Dgram* dg  = new(buffer) Dgram(*(event->creator()));
  void*  buf = dg->xtc.alloc(sz);
  memcpy(buf, event->begin(), sz);
  dg->env = (dg->env & 0xff00ffff) | (idx << 16); // Pass buffer's index to _deleteDatagram()

  if (_prms.verbose >= VL_EVENT)
  {
    uint64_t    pid = event->creator()->pulseId();
    unsigned    ctl = dg->control();
    unsigned    env = dg->env;
    size_t      sz  = sizeof(*dg) + dg->xtc.sizeofPayload();
    unsigned    src = dg->xtc.src.value();
    const char* svc = TransitionId::name(dg->service());
    printf("MEB processed %5lu %15s  [%8u] @ "
           "%16p, ctl %02x, pid %014lx, env %08x, sz %6zd, src %2u, ts %u.%09u\n",
           _eventCount, svc, idx, dg, ctl, pid, env, sz, src, dg->time.seconds(), dg->time.nanoseconds());
  }

  if (_apps->events(dg) == XtcMonitorServer::Handled)
  {
    // Make the buffer available to the contributor again
    post(event->begin(), event->end());

    Pool::free((void*)dg);          // Handled means _deleteDatagram() won't be called
  }
}


static std::string getHostname()
{
  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);
  return std::string(hostname);
}

class MebApp : public CollectionApp
{
public:
  MebApp(const std::string& collSrv, MebParams&);
  ~MebApp();
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
  void _shutdown();
  int  _parseConnectionParams(const json& msg);
  void _printParams(const EbParams& prms, unsigned groups) const;
  void _printGroups(unsigned groups, const u64arr_t& array) const;
private:
  MebParams&                           _prms;
  const bool                           _ebPortEph;
  std::unique_ptr<prometheus::Exposer> _exposer;
  std::shared_ptr<MetricExporter>      _exporter;
  std::unique_ptr<Meb>                 _meb;
  std::thread                          _appThread;
  uint16_t                             _groups;
  bool                                 _unconfigFlag;
};

MebApp::MebApp(const std::string& collSrv,
               MebParams&         prms) :
  CollectionApp(collSrv, prms.partition, "meb", prms.alias),
  _prms        (prms),
  _ebPortEph   (prms.ebPort.empty()),
  _exposer     (Pds::createExposer(prms.prometheusDir, getHostname())),
  _exporter    (std::make_shared<MetricExporter>()),
  _meb         (std::make_unique<Meb>(_prms, _exporter)),
  _unconfigFlag(false)
{
  if (_exposer)
  {
    _exposer->RegisterCollectable(_exporter);
  }

  logging::info("Ready for transitions");
}

MebApp::~MebApp()
{
  // Try to take things down gracefully when an exception takes us off the
  // normal path so that the most chance is given for prints to show up
  handleReset(json({}));
}

std::string MebApp::_error(const json&        msg,
                           const std::string& errorMsg)
{
  json body = json({});
  const std::string& key = msg["header"]["key"];
  body["err_info"] = errorMsg;
  logging::error("%s", errorMsg.c_str());
  reply(createMsg(key, msg["header"]["msg_id"], getId(), body));
  return errorMsg;
}

json MebApp::connectionInfo()
{
  // Allow the default NIC choice to be overridden
  if (_prms.ifAddr.empty())
  {
    _prms.ifAddr = _prms.kwargs.find("ep_domain") != _prms.kwargs.end()
                 ? getNicIp(_prms.kwargs["ep_domain"])
                 : getNicIp(_prms.kwargs["forceEnet"] == "yes");
  }

  // If port is not user specified, reset the previously allocated port number
  if (_ebPortEph)            _prms.ebPort.clear();

  int rc = _meb->startConnection(_prms.ifAddr, _prms.ebPort, MAX_DRPS);
  if (rc)  throw "Error starting connection";

  json body = {{"connect_info", {{"nic_ip",    _prms.ifAddr},
                                 {"meb_port",  _prms.ebPort},
                                 {"buf_count", _prms.numEvBuffers}}}};
  return body;
}

void MebApp::handleConnect(const json &msg)
{
  json body = json({});
  int  rc   = _parseConnectionParams(msg["body"]);
  if (rc)
  {
    _error(msg, "Error parsing connection parameters");
    return;
  }

  _meb->resetCounters();

  rc = _meb->connect();
  if (rc)
  {
    _error(msg, "Error in MEB connect()");
    return;
  }

  // Reply to collection with transition status
  reply(createMsg("connect", msg["header"]["msg_id"], getId(), body));
}

int MebApp::_configure(const json &msg)
{
  int rc = _meb->configure();
  if (rc)  logging::error("%s:\n  Failed to configure MEB",
                          __PRETTY_FUNCTION__);

  return rc;
}

void MebApp::_unconfigure()
{
  // Shut down the previously running instance, if any
  lRunning = 0;
  if (_appThread.joinable())  _appThread.join();

  _meb->unconfigure();
}

void MebApp::_shutdown()
{
  // Carry out the queued Unconfigure, if there was one
  if (_unconfigFlag)
  {
    _unconfigure();
    _unconfigFlag = false;
  }

  _meb->disconnect();
}

void MebApp::handlePhase1(const json& msg)
{
  json        body = json({});
  std::string key  = msg["header"]["key"];

  if (key == "configure")
  {
    // Carry out the queued Unconfigure, if any
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

    _appThread = std::thread(&Meb::run, std::ref(*_meb));
  }
  else if (key == "unconfigure")
  {
    // "Queue" unconfiguration until after phase 2 has completed
    _unconfigFlag = true;
  }
  else if (key == "beginrun")
  {
    if (_meb->beginrun())
    {
      _error(msg, "Phase 1 error: Failed to " + key);
      return;
    }
  }

  // Reply to collection with transition status
  reply(createMsg(key, msg["header"]["msg_id"], getId(), body));
}

void MebApp::handleDisconnect(const json &msg)
{
  _shutdown();

  // Reply to collection with connect status
  json body = json({});
  reply(createMsg("disconnect", msg["header"]["msg_id"], getId(), body));
}

void MebApp::handleReset(const json &msg)
{
  _unconfigFlag = true;
  _shutdown();
}

int MebApp::_parseConnectionParams(const json& body)
{
  std::string id = std::to_string(getId());
  _prms.id       = body["meb"][id]["meb_id"];
  if (_prms.id >= MAX_MEBS)
  {
    logging::error("MEB ID %u is out of range 0 - %u", _prms.id, MAX_MEBS - 1);
    return 1;
  }

  if (body.find("drp") == body.end())
  {
    logging::error("Missing required DRP specs");
    return 1;
  }

  size_t   maxTrSize     = 0;
  size_t   maxBufferSize = 0;
  _prms.contributors     = 0;
  _prms.maxBufferSize    = 0;
  _prms.maxTrSize.resize(body["drp"].size());

  _prms.contractors.fill(0);
  _prms.receivers.fill(0);
  _groups = 0;

  for (auto it : body["drp"].items())
  {
    unsigned drpId = it.value()["drp_id"];
    if (drpId > MAX_DRPS - 1)
    {
      logging::error("DRP ID %u is out of range 0 - %u", drpId, MAX_DRPS - 1);
      return 1;
    }
    _prms.contributors |= 1ul << drpId;

    _prms.addrs.push_back(it.value()["connect_info"]["nic_ip"]);
    _prms.ports.push_back(it.value()["connect_info"]["drp_port"]);

    unsigned group = it.value()["det_info"]["readout"];
    if (group > NUM_READOUT_GROUPS - 1)
    {
      logging::error("Readout group %u is out of range 0 - %u", group, NUM_READOUT_GROUPS - 1);
      return 1;
    }
    _prms.contractors[group] |= 1ul << drpId;
    _prms.receivers[group]    = 0;      // Unused by MEB
    _groups |= 1 << group;

    _prms.maxTrSize[drpId] = size_t(it.value()["connect_info"]["max_tr_size"]);
    maxTrSize             += _prms.maxTrSize[drpId];
    maxBufferSize         += size_t(it.value()["connect_info"]["max_ev_size"]);
  }
  // shmem buffers must fit both built events and transitions of worst case size
  _prms.maxBufferSize = maxBufferSize > maxTrSize ? maxBufferSize : maxTrSize;

  if (body.find("teb") == body.end())
  {
    logging::error("Missing required TEB specs");
    return 1;
  }

  _prms.addrs.clear();
  _prms.ports.clear();

  for (auto it : body["teb"].items())
  {
    unsigned tebId = it.value()["teb_id"];
    if (tebId > MAX_TEBS - 1)
    {
      logging::error("TEB ID %u is out of range 0 - %u", tebId, MAX_TEBS - 1);
      return 1;
    }
    _prms.addrs.push_back(it.value()["connect_info"]["nic_ip"]);
    _prms.ports.push_back(it.value()["connect_info"]["mrq_port"]);
  }

  return 0;
}

void MebApp::_printGroups(unsigned groups, const u64arr_t& array) const
{
  while (groups)
  {
    unsigned group = __builtin_ffs(groups) - 1;
    groups &= ~(1 << group);

    printf("%u: 0x%016lx  ", group, array[group]);
  }
  printf("\n");
}

void MebApp::_printParams(const EbParams& prms, unsigned groups) const
{
  printf("\nParameters of MEB ID %u (%s:%s):\n",               _prms.id,
                                                               _prms.ifAddr.c_str(), _prms.ebPort.c_str());
  printf("  Thread core numbers:        %d, %d\n",             _prms.core[0], _prms.core[1]);
  printf("  Partition:                  %u\n",                 _prms.partition);
  printf("  Bit list of contributors:   0x%016lx, cnt: %zd\n", _prms.contributors,
                                                               std::bitset<64>(_prms.contributors).count());
  printf("  Readout group contractors:  ");                    _printGroups(_groups, _prms.contractors);
  printf("  Number of TEB requestees:   %zd\n",                _prms.addrs.size());
  printf("  Buffer duration:            0x%014lx\n",           BATCH_DURATION);
  printf("  Number of event buffers:    0x%08x = %u\n",        _prms.numEvBuffers, _prms.numEvBuffers);
  printf("  Max # of entries / buffer:  0x%08x = %u\n",        1, 1);
  printf("  shmem buffer size:          0x%08x = %u\n",        _prms.maxBufferSize, _prms.maxBufferSize);
  printf("  Number of event queues:     0x%08x = %u\n",        _prms.nevqueues, _prms.nevqueues);
  printf("  Distribute:                 %s\n",                 _prms.ldist ? "yes" : "no");
  printf("  Tag:                        %s\n",                 _prms.tag.c_str());
  printf("\n");
}


using namespace Pds;


static
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
                  "[-M <Prometheus config file directory>]"
                  "[-k <keyword arguments>]"
                  "[-v] "
                  "[-h] "
                  "\n", progname);
}

int main(int argc, char** argv)
{
  const unsigned NO_PARTITION = unsigned(-1u);
  std::string    collSrv;
  MebParams      prms;
  std::string    kwargs_str;

  prms.partition = NO_PARTITION;
  prms.core[0]   = CORE_0;
  prms.core[1]   = CORE_1;
  prms.verbose   = 0;

  prms.maxBufferSize = 0;     // Filled in @ connect
  prms.numEvBuffers  = NUMBEROF_XFERBUFFERS;
  prms.nevqueues     = 1;
  prms.ldist         = false;

  int c;
  while ((c = getopt(argc, argv, "p:P:n:t:q:dA:C:1:2:u:M:k:vh")) != -1)
  {
    errno = 0;
    char* endPtr;
    switch (c) {
      case 'p':
        prms.partition = strtoul(optarg, &endPtr, 0);
        if (errno != 0 || endPtr == optarg) prms.partition = NO_PARTITION;
        break;
      case 'P':
        prms.instrument = std::string(optarg);
        break;
      case 'n':
        sscanf(optarg, "%u", &prms.numEvBuffers);
        break;
      case 't':
        prms.tag = std::string(optarg);
        break;
      case 'q':
        prms.nevqueues = strtoul(optarg, NULL, 0);
        break;
      case 'd':
        prms.ldist = true;
        break;
      case 'A':  prms.ifAddr        = optarg;                      break;
      case 'C':  collSrv            = optarg;                      break;
      case '1':  prms.core[0]       = atoi(optarg);                break;
      case '2':  prms.core[1]       = atoi(optarg);                break;
      case 'u':  prms.alias         = optarg;                      break;
      case 'M':  prms.prometheusDir = optarg;                      break;
      case 'k':  kwargs_str         = kwargs_str.empty()
                                    ? optarg
                                    : kwargs_str + ", " + optarg;  break;
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

  logging::init(prms.instrument.c_str(), prms.verbose ? LOG_DEBUG : LOG_INFO);
  logging::info("logging configured");

  if (prms.partition == NO_PARTITION)
  {
    logging::critical("-p: partition number is mandatory");
    return 1;
  }
  if (prms.instrument.empty())
  {
    logging::critical("-P: instrument name is mandatory");
    return 1;
  }
  if (!prms.numEvBuffers)
  {
    logging::critical("-n: max buffers is mandatory");
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

  if (prms.numEvBuffers < NUMBEROF_XFERBUFFERS)
    prms.numEvBuffers = NUMBEROF_XFERBUFFERS;
  if (prms.numEvBuffers > 255)
  {
    // The problem is that there are only 8 bits available in the env
    // Could use the lower 24 bits, but then we have a nonstandard env
    logging::critical("%s:\n  Number of event buffers > 255 is not supported: got %u", prms.numEvBuffers);
    return 1;
  }

  if (prms.tag.empty())  prms.tag = prms.instrument;
  logging::info("Partition Tag: '%s'", prms.tag.c_str());

  get_kwargs(kwargs_str, prms.kwargs);

  struct sigaction sigAction;

  sigAction.sa_handler = sigHandler;
  sigAction.sa_flags   = SA_RESTART;
  sigemptyset(&sigAction.sa_mask);
  if (sigaction(SIGINT, &sigAction, &lIntAction) > 0)
    logging::warning("Failed to set up ^C handler");

  try
  {
    MebApp app(collSrv, prms);

    app.run();

    return 0;
  }
  catch (std::exception& e)  { logging::critical("%s", e.what()); }
  catch (std::string& e)     { logging::critical("%s", e.c_str()); }
  catch (char const* e)      { logging::critical("%s", e); }
  catch (...)                { logging::critical("Default exception"); }

  return EXIT_FAILURE;
}
