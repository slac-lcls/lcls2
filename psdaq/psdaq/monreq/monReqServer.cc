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
#include "psdaq/service/fast_monotonic_clock.hh"
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
#include <chrono>
#include <mutex>

#define UNLIKELY(expr)  __builtin_expect(!!(expr), 0)
#define LIKELY(expr)    __builtin_expect(!!(expr), 1)

#ifndef POSIX_TIME_AT_EPICS_EPOCH
#define POSIX_TIME_AT_EPICS_EPOCH 631152000u
#endif

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
using tp_t     = std::chrono::system_clock::time_point;
using ms_t     = std::chrono::milliseconds;
using us_t     = std::chrono::microseconds;
using ns_t     = std::chrono::nanoseconds;

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


  class MyMetric
  {
  public:
    MyMetric(unsigned num) :
      _cnt(0ull),
      _t0(num),
      _dt(0ull),
      _minT(ULLONG_MAX),
      _maxT(0ull),
      _tBeg(0ull),
      _cBeg(0ull)
    {}
#define LCK std::lock_guard<std::mutex> lk(_lock)
    void start(unsigned idx)
    {
      LCK;
      _t0[idx] = fast_monotonic_clock::now(CLOCK_MONOTONIC);
    }
    void accumulate(unsigned idx)
    {
      if (_t0[idx] != _EPOCH)           // Avoid large startup values
      {
        LCK;
        ++_cnt;
        auto     now = fast_monotonic_clock::now(CLOCK_MONOTONIC);
        uint64_t dt  = std::chrono::duration_cast<ns_t>(now - _t0[idx]).count();
        _dt    = dt;
        _accT += dt;
        if (dt < _minT)  _minT = dt;
        if (dt > _maxT)  _maxT = dt;
      }
    }
    uint64_t count()   { return _cnt; }
    uint64_t sample()  { return _dt; }
    uint64_t minimum() { LCK; auto minT = _minT;  _minT = ULLONG_MAX;  return minT; }
    uint64_t maximum() { LCK; auto maxT = _maxT;  _maxT = 0ull;        return maxT; }
    uint64_t average()
    {
      LCK;
      auto dt = _accT - _tBeg;  _tBeg = _accT;
      auto dc = _cnt  - _cBeg;  _cBeg = _cnt;
      return dc ? (dt / dc) : 0ull;
    }
#undef LCK
  private:
    using tp_t = std::chrono::time_point<fast_monotonic_clock>;
    const tp_t         _EPOCH;
    mutable std::mutex _lock;
    uint64_t           _cnt;
    std::vector<tp_t>  _t0;
    uint64_t           _dt;
    uint64_t           _accT;
    uint64_t           _minT;
    uint64_t           _maxT;
    uint64_t           _tBeg;
    uint64_t           _cBeg;
  };

  class MyXtcMonitorServer : public XtcMonitorServer {
  public:
    MyXtcMonitorServer(std::vector<EbLfCltLink*>&     links,
                       uint64_t&                      requestCount,
                       std::shared_ptr<PromHistogram> bufUseCnts,
                       std::atomic<uint64_t>&         prcBufCount,
                       MyMetric&                      bufPrcMetric,
                       MyMetric&                      monTrgMetric,
                       MyMetric&                      appPrcMetric,
                       const MebParams&               prms) :
      XtcMonitorServer(prms.tag.c_str(),
                       prms.maxBufferSize,
                       prms.numEvBuffers,
                       prms.nevqueues),
      _mrqLinks    (links),
      _requestCount(requestCount),
      _bufFreeList (prms.numEvBuffers),
      _bufUseCnts  (bufUseCnts),
      _prcBufCount (prcBufCount),
      _bufPrcMetric(bufPrcMetric),
      _monTrgMetric(monTrgMetric),
      _appPrcMetric(appPrcMetric),
      _prms        (prms)
    {
      for (unsigned i = 0; i < _bufFreeList.size(); ++i)
      {
        if (_bufFreeList.push(i))
        {
          logging::critical("%s:\n  _bufFreeList.push(%u) failed", __PRETTY_FUNCTION__, i);
          abort();
        }
      }

      // Clear the counter here because _init() will cause it to count
      _requestCount = 0;
      _bufUseCnts->clear();

      _init();
    }
    virtual ~MyXtcMonitorServer()
    {
    }
    const size_t bufListCount() const
    {
      return _bufFreeList.count();
    }
  private:
    virtual void _copyDatagram(Dgram* dg, char* buf, size_t bSz)
    {
      if (UNLIKELY(_prms.verbose >= VL_EVENT))
        fprintf(stderr, "_copyDatagram:   dg %p, ts %u.%09u to %p\n",
                dg, dg->time.seconds(), dg->time.nanoseconds(), buf);

      // The dg payload is a directory of contributions to the built event.
      // Iterate over the directory and construct, in shared memory, the event
      // datagram (odg) from the contribution XTCs
      const EbDgram** const  last = (const EbDgram**)dg->xtc.next();
      const EbDgram*  const* ctrb = (const EbDgram**)dg->xtc.payload();
      Dgram*                 odg  = new((void*)buf) Dgram(**ctrb); // Not an EbDgram!
      odg->xtc.src      = XtcData::Src(_prms.id, XtcData::Level::Event);
      odg->xtc.contains = XtcData::TypeId(XtcData::TypeId::Parent, 0);
      const void* bufEnd = buf + bSz;
      do
      {
        const EbDgram* idg = *ctrb;

        odg->xtc.damage.increase(idg->xtc.damage.value());

        size_t   oSz  = sizeof(*odg) + odg->xtc.sizeofPayload();
        uint32_t iExt = idg->xtc.extent;
        // Truncation goes unnoticed, so crash instead to get it fixed
        if (oSz + iExt > bSz)
        {
          logging::critical("Buffer of size %zu (%zu in use) is too small to add Xtc of size %zu",
                            bSz, oSz, iExt);
          throw "Buffer too small";
        }
        buf = (char*)odg->xtc.alloc(iExt, bufEnd);
        memcpy(buf, &idg->xtc, iExt);
      }
      while (++ctrb != last);
    }

    virtual void _deleteDatagram(Dgram* dg) // Not called for transitions
    {
      unsigned idx = dg->xtc.src.value();

      if (UNLIKELY(_prms.verbose >= VL_EVENT))
        fprintf(stderr, "_deleteDatagram: dg %p, ts %u.%09u, idx %u\n",
                dg, dg->time.seconds(), dg->time.nanoseconds(), idx);

      if (idx >= _bufFreeList.size())
      {
        logging::warning("deleteDatagram: Out of bounds index %08x, max %08x",
                         idx, _bufFreeList.size() - 1);
      }

      for (unsigned i = 0; i < _bufFreeList.count(); ++i)
      {
        if (idx == _bufFreeList.peek(i))
        {
          logging::error("Index is already on list at %u: idx %u, dg %p, ts %u.%09u, svc %s",
                         i, idx, dg, dg->time.seconds(), dg->time.nanoseconds(), TransitionId::name(dg->service()));
          //Pool::free((void*)dg);
          return;
        }
      }
      // Number of buffers being processed by the MEB; incremented in Meb::process()
      --_prcBufCount;                   // L1Accepts only
      _appPrcMetric.start(idx);
      _bufPrcMetric.accumulate(idx);

      if (_bufFreeList.push(idx))
      {
        logging::error("_bufFreeList.push(%u) failed, count %zd", idx, _bufFreeList.count());
        for (unsigned i = 0; i < _bufFreeList.size(); ++i)
        {
          fprintf(stderr, "Free list entry %u: %u\n", i, _bufFreeList.peek(i));
        }
      }
      //printf("_deleteDatagram: push idx %u, cnt = %zu\n", idx, _bufFreeList.count());

      Pool::free((void*)dg);
    }

    virtual void _requestDatagram()
    {
      //printf("_requestDatagram\n");

      unsigned idx;
      if (_bufFreeList.pop(idx))
      {
        logging::error("%s:\n  No free buffers available", __PRETTY_FUNCTION__);
        return;
      }
      //printf("_requestDatagram: pop idx %u, cnt = %zu\n", data, _bufFreeList.count());

      auto data = ImmData::value(ImmData::Buffer, _prms.id, idx);

      // Split the pool of indices across all TEBs evenly
      unsigned iTeb = idx % _mrqLinks.size();
      int rc = _mrqLinks[iTeb]->EbLfLink::post(data);
      if (rc == 0)
      {
        ++_requestCount;
        _bufUseCnts->observe(double(idx));
        _monTrgMetric.start(idx);
        _appPrcMetric.accumulate(idx);
      }
      else
      {
        logging::error("%s:\n  Unable to post request to TEB %u: rc %d, idx %u (%08x)",
                       __PRETTY_FUNCTION__, iTeb, rc, idx, data);

        // Don't leak buffers - Revisit: XtcMonServer leaks in this case
        if (_bufFreeList.push(idx))
        {
          logging::error("_bufFreeList.push(%u) failed, count %zd", idx, _bufFreeList.count());
          for (unsigned i = 0; i < _bufFreeList.size(); ++i)
          {
            fprintf(stderr, "Free list entry %u: %u\n", i, _bufFreeList.peek(i));
          }
        }
      }

      if (UNLIKELY(_prms.verbose >= VL_EVENT))
        fprintf(stderr, "_requestDatagram: Post EB[iTeb %u], value %08x, rc %d\n",
                iTeb, data, rc);
    }

  private:
    std::vector<EbLfCltLink*>&     _mrqLinks;
    uint64_t&                      _requestCount;
    FifoMT<unsigned, std::mutex>   _bufFreeList;
    std::shared_ptr<PromHistogram> _bufUseCnts;
    std::atomic<uint64_t>&         _prcBufCount;
    MyMetric&                      _bufPrcMetric;
    MyMetric&                      _monTrgMetric;
    MyMetric&                      _appPrcMetric;
    const MebParams&               _prms;
  };


  class Meb : public EbAppBase
  {
  public:
    Meb(const MebParams& prms, ZmqContext& context);
  public:
    int  resetCounters();
    int  connect(const std::shared_ptr<MetricExporter>);
    int  configure();
    void unconfigure();
    void disconnect();
    void shutdown();
    void run();
  private:                              // For EventBuilder
    virtual
    void process(EbEvent* event);
  private:
      int _setupMetrics(const std::shared_ptr<MetricExporter>);
  private:
    std::unique_ptr<MyXtcMonitorServer> _apps;
    std::vector<EbLfCltLink*>           _mrqLinks;
    std::unique_ptr<GenericPool>        _pool;
    uint64_t                            _pidPrv;
    uint64_t                            _latPid;
    int64_t                             _latency;
    uint64_t                            _eventCount;
    uint64_t                            _trCount;
    uint64_t                            _splitCount;
    uint64_t                            _requestCount;
    std::atomic<uint64_t>               _prcBufCount;
    std::shared_ptr<PromHistogram>      _bufUseCnts;
    MyMetric                            _bufPrcMetric;
    MyMetric                            _monTrgMetric;
    MyMetric                            _appPrcMetric;
    uint64_t                            _scrapeCount;
    std::vector<uint64_t>               _rogCount;
    const MebParams&                    _prms;
    EbLfClient                          _mrqTransport;
    ZmqSocket                           _inprocSend;
  };
};


static json createPulseIdMsg(uint64_t pulseId)
{
  json msg, body;
  msg["key"] = "pulseId";
  body["pulseId"] = pulseId;
  msg["body"] = body;
  return msg;
}

Meb::Meb(const MebParams&        prms,
         ZmqContext&             context) :
  EbAppBase    (prms, "MEB"),
  _pidPrv      (0),
  _latPid      (0),
  _latency     (0),
  _eventCount  (0),
  _trCount     (0),
  _splitCount  (0),
  _requestCount(0),
  _prcBufCount (0),
  _bufPrcMetric(prms.numEvBuffers),
  _monTrgMetric(prms.numEvBuffers),
  _appPrcMetric(prms.numEvBuffers),
  _scrapeCount (0),
  _rogCount    (NUM_READOUT_GROUPS, 0),
  _prms        (prms),
  _mrqTransport(prms.verbose, prms.kwargs),
  _inprocSend  (&context, ZMQ_PAIR)
{
  _inprocSend.connect("inproc://drp");  // Yes, 'drp' is the name
}

int Meb::resetCounters()
{
  EbAppBase::resetCounters();

  _latency    = 0;
  _eventCount = 0;
  _trCount    = 0;
  _splitCount = 0;

  return 0;
}

void Meb::shutdown()
{
  // If connect() ran but the system didn't get into the Connected state,
  // there won't be a Disconnect transition, so disconnect() here
  disconnect();                         // Does no harm if already done

  EbAppBase::shutdown();
}

void Meb::disconnect()
{
  // If configure() ran but the system didn't get into the Configured state,
  // there won't be an Unconfigure transition, so unconfigure() here
  unconfigure();                        // Does no harm if already done

  for (auto link : _mrqLinks)  _mrqTransport.disconnect(link);
  _mrqLinks.clear();

  EbAppBase::disconnect();
}

void Meb::unconfigure()
{
  if (_pool)
  {
    fprintf(stderr, "Directory datagram pool:\n");
    _pool->dump();
  }
  _pool.reset();

  _apps.reset();

  EbAppBase::unconfigure();
}

int Meb::_setupMetrics(const MetricExporter_t exporter)
{
  std::map<std::string, std::string> labels{{"instrument", _prms.instrument},
                                            {"partition", std::to_string(_prms.partition)},
                                            {"detname", _prms.alias},
                                            {"alias", _prms.alias}};
  exporter->add("MEB_EvtCt",  labels, MetricType::Counter, [&](){ return _eventCount;      });
  exporter->add("MEB_TrCt",   labels, MetricType::Counter, [&](){ return _trCount;         });
  exporter->add("MEB_EvtLat", labels, MetricType::Gauge,   [&](){ return _latency;         });
  exporter->add("MEB_SpltCt", labels, MetricType::Counter, [&](){ return _splitCount;      });
  exporter->add("MEB_ReqCt",  labels, MetricType::Counter, [&](){ return _requestCount;    });
  exporter->add("MRQ_TxPdg",  labels, MetricType::Gauge,   [&](){ return _mrqTransport.posting(); });
  exporter->add("MRQ_BufCt",  labels, MetricType::Gauge,   [&](){ return _apps ? _apps->bufListCount() : 0; });
  exporter->add("MEB_PrcCt",  labels, MetricType::Gauge,   [&](){ return _prcBufCount.load(); });
  exporter->add("MEB_PrcTmC", labels, MetricType::Gauge,   [&](){ return _bufPrcMetric.count();   });
  exporter->add("MEB_PrcTm",  labels, MetricType::Gauge,   [&](){ return _bufPrcMetric.sample();  });
  exporter->add("MEB_PrcTmm", labels, MetricType::Gauge,   [&](){ return _bufPrcMetric.minimum(); });
  exporter->add("MEB_PrcTmM", labels, MetricType::Gauge,   [&](){ return _bufPrcMetric.maximum(); });
  exporter->add("MEB_PrcTmA", labels, MetricType::Gauge,   [&](){ return _bufPrcMetric.average(); });
  exporter->add("MEB_TrgTmC", labels, MetricType::Gauge,   [&](){ return _monTrgMetric.count();   });
  exporter->add("MEB_TrgTm",  labels, MetricType::Gauge,   [&](){ return _monTrgMetric.sample();  });
  exporter->add("MEB_TrgTmm", labels, MetricType::Gauge,   [&](){ return _monTrgMetric.minimum(); });
  exporter->add("MEB_TrgTmM", labels, MetricType::Gauge,   [&](){ return _monTrgMetric.maximum(); });
  exporter->add("MEB_TrgTmA", labels, MetricType::Gauge,   [&](){ return _monTrgMetric.average(); });
  exporter->add("MEB_AppTmC", labels, MetricType::Gauge,   [&](){ return _appPrcMetric.count();   });
  exporter->add("MEB_AppTm",  labels, MetricType::Gauge,   [&](){ return _appPrcMetric.sample();  });
  exporter->add("MEB_AppTmm", labels, MetricType::Gauge,   [&](){ return _appPrcMetric.minimum(); });
  exporter->add("MEB_AppTmM", labels, MetricType::Gauge,   [&](){ return _appPrcMetric.maximum(); });
  exporter->add("MEB_AppTmA", labels, MetricType::Gauge,   [&](){ return _appPrcMetric.average(); });
  exporter->add("MEB_ScrpCt", labels, MetricType::Counter, [&](){ return _scrapeCount++;  });
  exporter->add("MEB_RogCt0", labels, MetricType::Counter, [&](){ return _rogCount[0];    });
  exporter->add("MEB_RogCt1", labels, MetricType::Counter, [&](){ return _rogCount[1];    });
  exporter->add("MEB_RogCt2", labels, MetricType::Counter, [&](){ return _rogCount[2];    });
  exporter->add("MEB_RogCt3", labels, MetricType::Counter, [&](){ return _rogCount[3];    });
  exporter->add("MEB_RogCt4", labels, MetricType::Counter, [&](){ return _rogCount[4];    });
  exporter->add("MEB_RogCt5", labels, MetricType::Counter, [&](){ return _rogCount[5];    });
  exporter->add("MEB_RogCt6", labels, MetricType::Counter, [&](){ return _rogCount[6];    });
  exporter->add("MEB_RogCt7", labels, MetricType::Counter, [&](){ return _rogCount[7];    });
  exporter->constant("MRQ_BufCtMax", labels, _prms.numEvBuffers);
  _bufUseCnts = exporter->histogram("MRQ_BufUseCnts", labels, _prms.numEvBuffers);

  return 0;
}

int Meb::connect(const std::shared_ptr<MetricExporter> exporter)
{
  if (exporter)
  {
    int rc = _setupMetrics(exporter);
    if (rc)  return rc;
  }

  _mrqLinks.resize(_prms.addrs.size());

  int rc = linksConnect(_mrqTransport, _mrqLinks, _prms.addrs, _prms.ports, _prms.id, "TEB");
  if (rc)  return rc;

  rc = EbAppBase::connect(MEB_TR_BUFFERS, exporter);
  if (rc)  return rc;

  return 0;
}

int Meb::configure()
{
  // Create pool for transferring events to MyXtcMonitorServer
  unsigned entries = std::bitset<64>(_prms.contributors).count();
  size_t   size    = sizeof(Dgram) + entries * sizeof(Dgram*);
  _pool = std::make_unique<GenericPool>(size, 1 + _prms.numEvBuffers); // +1 for Transitions

  // MRQ links need no configuration

  int rc = EbAppBase::configure();
  if (rc)  return rc;

  // Code added here involving the links must be coordinated with the other side

  _apps = std::make_unique<MyXtcMonitorServer>(_mrqLinks, _requestCount,
                                               _bufUseCnts, _prcBufCount,
                                               _bufPrcMetric, _monTrgMetric, _appPrcMetric,
                                               _prms);

  _apps->distribute(_prms.ldist);

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

  int rcPrv = 0;
  while (lRunning)
  {
    rc = EbAppBase::process();
    if (rc < 0)
    {
      if (rc == -FI_EAGAIN)
      {
        rc = 0;
      }
      else if (rc == -FI_ENOTCONN)
      {
        logging::critical("MEB thread lost connection with a DRP");
        throw "Receiver thread lost connection with a DRP";
      }
      else if (rc == rcPrv)
      {
        logging::critical("MEB thread aborting on repeating fatal error");
        throw "Repeating fatal error";
      }
    }
    rcPrv = rc;
  }

  EventBuilder::dump(0);

  logging::info("MEB thread finished");
}

void Meb::process(EbEvent* event)
{
  if (_prms.verbose >= VL_DETAILED)
  {
    fprintf(stderr, "Meb::process event dump:\n");
    event->dump(1, _trCount + _eventCount);
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
  if (UNLIKELY(!(pid > _pidPrv)))
  {
    event->damage(Damage::OutOfOrder);

    logging::critical("%s:\n  Pulse ID did not advance: %014lx <= %014lx, rem %08lx, prm %08x, svc %u, ts %u.%09u",
                      __PRETTY_FUNCTION__, pid, _pidPrv, event->remaining(), event->immData(), dgram->service(), dgram->time.seconds(), dgram->time.nanoseconds());

    if (event->remaining())             // I.e., this event was fixed up
    {
      // This can happen only for a split event (I think), which was fixed up and
      // posted earlier, so return to dismiss this counterpart and not post it
      // However, we can't know whether this is a split event or a fixed-up out-of-order event
      ++_splitCount;
      logging::critical("%s:\n  Split event, if pid %014lx was fixed up multiple times",
                        __PRETTY_FUNCTION__, pid);
      // return, if we knew this PID had been fixed up before
    }
    throw "Pulse ID did not advance";   // Can't recover from non-spit events
  }
  _pidPrv = pid;

  if (dgram->isEvent())  ++_eventCount;
  else                   ++_trCount;

  // Create a Dgram with a payload that is a directory of contribution
  // Dgrams to the built event in order to avoid first assembling the
  // datagram from its contributions in a temporary buffer, only to then
  // copy it into shmem in _copyDatagram() above.  Since the contributions,
  // and thus the full datagram, can be quite large, this would amount to
  // a lot of copying
  size_t      sz     = (event->end() - event->begin()) * sizeof(*(event->begin()));
  unsigned    idx    = ImmData::idx(event->immData());
  Dgram*      dg     = new(_pool->alloc(sizeof(Dgram) + sz)) Dgram(*(event->creator()));
  const void* bufEnd = ((char*)dg) + _pool->sizeofObject();
  if (!dg)
  {
    logging::critical("%s:\n  Dgram pool allocation of size %zd failed:",
                      __PRETTY_FUNCTION__, sizeof(Dgram) + sz);
    fprintf(stderr, "Directory datagram pool\n");
    _pool->dump();
    fprintf(stderr, "Meb::process event dump:\n");
    event->dump(1, -1);
    abort();
  }

  dg->xtc.src = {idx, Level::Event}; // Pass buffer's index to _deleteDatagram()

  void*  buf = dg->xtc.alloc(sz, bufEnd);
  memcpy(buf, event->begin(), sz);

  if (_prms.verbose >= VL_EVENT)
  {
    unsigned    ctl = dg->control();
    unsigned    env = dg->env;
    size_t      sz  = sizeof(*dg) + dg->xtc.sizeofPayload();
    unsigned    src = dg->xtc.src.value();
    const char* svc = TransitionId::name(dg->service());
    auto        cnt = dg->isEvent() ? _eventCount : _trCount;
    fprintf(stderr, "MEB processed %5lu %15s  [%8u] @ "
            "%16p, ctl %02x, pid %014lx, env %08x, sz %6zd, src %2u, ts %u.%09u\n",
            cnt, svc, idx, dg, ctl, pid, env, sz, src, dg->time.seconds(), dg->time.nanoseconds());
  }
  else
  {
    auto svc = dg->service();
    if ((svc != TransitionId::L1Accept) && (svc != TransitionId::SlowUpdate))
    {
      logging::info("MEB built      %15s @ %u.%09u (%014lx) from buffer %2u @ %16p",
                    TransitionId::name(svc),
                    dg->time.seconds(), dg->time.nanoseconds(),
                    pid, dg->xtc.src.value(), dg);
    }
  }

  if (!dgram->isEvent() || (dgram->pulseId() - _latPid > 13000000/14)) {
    _latency = latency<us_t>(dgram->time);
    _latPid  = dgram->pulseId();
  }

  if (dg->service() == TransitionId::L1Accept)
  {
    ++_prcBufCount;    // Number of buffers being processed by the MEB; decremented in _deleteDatagram
    _bufPrcMetric.start(idx);
    _monTrgMetric.accumulate(idx);

    int env = dg->env & _prms.rogs;
    while (env)
    {
      auto rog = __builtin_ffs(env) - 1;
      env &= ~(1 << rog);
      ++_rogCount[rog];
    }
  }

  // Transitions, including SlowUpdates, return Handled, L1Accepts return Deferred
  if (_apps->events(dg) == XtcMonitorServer::Handled)
  {
    // Make the transition buffer available to the contributors again
    post(event->begin(), event->end());

    if (dg->service() != TransitionId::SlowUpdate)
    {
      // send pulseId to inproc so it gets forwarded to the collection
      json msg = createPulseIdMsg(pid);
      _inprocSend.send(msg.dump());
    }

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
  virtual ~MebApp();
public:                                 // For CollectionApp
  json connectionInfo(const json& msg) override;
  void connectionShutdown() override;
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
  void _printParams(const MebParams& prms) const;
private:
  MebParams&                           _prms;
  const bool                           _ebPortEph;
  std::unique_ptr<prometheus::Exposer> _exposer;
  std::shared_ptr<MetricExporter>      _exporter;
  std::unique_ptr<Meb>                 _meb;
  std::thread                          _appThread;
  bool                                 _unconfigFlag;
};

MebApp::MebApp(const std::string& collSrv,
               MebParams&         prms) :
  CollectionApp(collSrv, prms.partition, "meb", prms.alias),
  _prms        (prms),
  _ebPortEph   (prms.ebPort.empty()),
  _exposer     (createExposer(prms.prometheusDir, getHostname())),
  _meb         (std::make_unique<Meb>(_prms, context())),
  _unconfigFlag(false)
{
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

json MebApp::connectionInfo(const nlohmann::json& msg)
{
  // Allow the default NIC choice to be overridden
  if (_prms.ifAddr.empty())
  {
    _prms.ifAddr = _prms.kwargs.find("ep_domain") != _prms.kwargs.end()
                 ? getNicIp(_prms.kwargs["ep_domain"])
                 : getNicIp(_prms.kwargs["forceEnet"] == "yes");
  }

  // If port is not user specified, reset the previously allocated port number
  if (_ebPortEph)  _prms.ebPort.clear();

  int rc = _meb->startConnection(_prms.ifAddr, _prms.ebPort, MAX_DRPS);
  if (rc)  throw "Error starting connection";

  json body = {{"connect_info", {{"nic_ip",       _prms.ifAddr},
                                 {"meb_port",     _prms.ebPort},
                                 {"max_ev_count", _prms.numEvBuffers}}}};
  return body;
}

void MebApp::connectionShutdown()
{
  _meb->shutdown();
}

void MebApp::handleConnect(const json &msg)
{
  // If the exporter already exists, replace it so that previous metrics are deleted
  if (_exposer)
  {
    _exporter = std::make_shared<MetricExporter>();
    _exposer->RegisterCollectable(_exporter);
  }

  json body = json({});
  int  rc   = _parseConnectionParams(msg["body"]);
  if (rc)
  {
    _error(msg, "Connection parameters error - see log");
    return;
  }

  rc = _meb->connect(_exporter);
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

  _printParams(_prms);

  _meb->resetCounters();                // Same time as DRPs

  return rc;
}

void MebApp::_unconfigure()
{
  // Shut down the previously running instance, if any
  lRunning = 0;
  if (_appThread.joinable())  _appThread.join();

  _meb->unconfigure();

  _unconfigFlag = false;
}

void MebApp::handlePhase1(const json& msg)
{
  json        body = json({});
  std::string key  = msg["header"]["key"];

  if (key == "configure")
  {
    // Handle a "queued" Unconfigure, if any
    if (_unconfigFlag)  _unconfigure();

    int rc = _configure(msg);
    if (rc)
    {
      _error(msg, "Phase 1 error: Failed to " + key);
      return;
    }

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
    _meb->resetCounters();              // Same time as DRPs
  }

  // Reply to collection with transition status
  reply(createMsg(key, msg["header"]["msg_id"], getId(), body));
}

void MebApp::handleDisconnect(const json &msg)
{
  // Carry out the queued Unconfigure, if there was one
  if (_unconfigFlag)  _unconfigure();

  _meb->disconnect();

  if (_exporter)  _exporter.reset();

  // Reply to collection with connect status
  json body = json({});
  reply(createMsg("disconnect", msg["header"]["msg_id"], getId(), body));
}

void MebApp::handleReset(const json &msg)
{
  unsubscribePartition();               // ZMQ_UNSUBSCRIBE

  _unconfigure();
  _meb->disconnect();
  if (_exporter)  _exporter.reset();
  connectionShutdown();
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

  size_t maxTrSize     = 0;
  size_t maxBufferSize = 0;
  _prms.contributors   = 0;
  _prms.maxBufferSize  = 0;
  _prms.maxTrSize.resize(body["drp"].size());

  _prms.rogs = 0;
  _prms.contractors.fill(0);
  _prms.receivers.fill(0);

  _prms.maxEntries = 1;                  // No batching: each event stands alone
  _prms.maxBuffers = _prms.numEvBuffers; // For EbAppBase
  _prms.numBuffers.resize(MAX_DRPS, 0);  // Number of buffers on each DRP
  _prms.drps.resize(MAX_DRPS);           // DRP aliases

  for (auto it : body["drp"].items())
  {
    unsigned drpId = it.value()["drp_id"];
    if (drpId > MAX_DRPS - 1)
    {
      logging::error("DRP ID %u is out of range 0 - %u", drpId, MAX_DRPS - 1);
      return 1;
    }
    _prms.contributors |= 1ul << drpId;
    _prms.drps[drpId]   = it.value()["proc_info"]["alias"];

    _prms.addrs.push_back(it.value()["connect_info"]["nic_ip"]);
    _prms.ports.push_back(it.value()["connect_info"]["drp_port"]);

    unsigned rog = it.value()["det_info"]["readout"];
    if (rog > NUM_READOUT_GROUPS - 1)
    {
      logging::error("Readout group %u is out of range 0 - %u", rog, NUM_READOUT_GROUPS - 1);
      return 1;
    }
    _prms.rogs             |= 1 << rog;
    _prms.contractors[rog] |= 1ul << drpId;
    _prms.receivers[rog]    = 0;      // Unused by MEB

    _prms.numBuffers[drpId] = _prms.maxBuffers;
    _prms.indexSources      = ULLONG_MAX; // All DRPs provide an index in immData

    _prms.maxTrSize[drpId] = size_t(it.value()["connect_info"]["max_tr_size"]);
    maxTrSize             += _prms.maxTrSize[drpId];
    maxBufferSize         += size_t(it.value()["connect_info"]["max_ev_size"]);
  }
  _prms.drps.shrink_to_fit();

  // shmem buffers must fit both built events and transitions of worst case size
  _prms.maxBufferSize = maxBufferSize > maxTrSize ? maxBufferSize : maxTrSize;

  unsigned suRate(body["control"]["0"]["control_info"]["slow_update_rate"]);
  if (1000 * MEB_TR_BUFFERS < suRate * EB_TMO_MS)
  {
    // Adjust EB_TMO_MS, MEB_TR_BUFFERS (in eb.hh) or the SlowUpdate rate
    logging::error("Increase # of MEB transition buffers from %u to > %u "
                   "for %u Hz of SlowUpdates and %u ms EB timeout\n",
                   MEB_TR_BUFFERS, (suRate * EB_TMO_MS + 999) / 1000, suRate, EB_TMO_MS);
    return 1;
  }

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

static
void _printGroups(const char* which, unsigned groups, const u64arr_t& array)
{
  char buffer[8*24];
  int i = 0;

  groups &= 0xff;                       // Prevent buffer overrun

  while (groups)
  {
    unsigned group = __builtin_ffs(groups) - 1;
    groups &= ~(1 << group);

    i += snprintf(&buffer[i], sizeof(buffer), "%u: 0x%016lx  ", group, array[group]);
  }
  logging::info("Readout group %-12s    %s", which, buffer);
}

void MebApp::_printParams(const MebParams& prms) const
{
  logging::info("");
  logging::info("Parameters of MEB ID %u (%s:%s):",                 prms.id,
                                                                    prms.ifAddr.c_str(), prms.ebPort.c_str());
  logging::info("  Thread core numbers:        %d, %d",             prms.core[0], prms.core[1]);
  logging::info("  Instrument:                 %s",                 prms.instrument.c_str());
  logging::info("  Partition:                  %u",                 prms.partition);
  logging::info("  Alias:                      %s",                 prms.alias.c_str());
  logging::info("  Bit list of contributors:   0x%016lx, cnt: %zu", prms.contributors,
                                                                    std::bitset<64>(prms.contributors).count());
  _printGroups("contractors:", prms.rogs, prms.contractors);
  logging::info("  # of TEB requestees:        %zu",                prms.addrs.size());
  logging::info("  Buffer duration:            %u",                 prms.maxEntries);
  logging::info("  Max # of entries / buffer:  0x%08x = %u",        prms.maxEntries, prms.maxEntries);
  logging::info("  # of event      buffers:    0x%08x = %u",        prms.numEvBuffers, prms.numEvBuffers);
  logging::info("  # of transition buffers:    0x%08x = %u",        MEB_TR_BUFFERS, MEB_TR_BUFFERS);
  logging::info("  Max buffer size:            0x%08x = %u",        prms.maxBufferSize, prms.maxBufferSize);
  logging::info("  # of event message queues:  0x%08x = %u",        prms.nevqueues, prms.nevqueues);
  logging::info("  Distribute:                 %s",                 prms.ldist ? "yes" : "no");
  logging::info("  Tag:                        %s",                 prms.tag.c_str());
  logging::info("");
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

  prms.maxBufferSize = 0;               // Filled in @ connect
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
      case '?':
      case 'h':
      default:
        usage(argv[0]);
        return 1;
    }
  }

  logging::init(prms.instrument.c_str(), prms.verbose ? LOG_DEBUG : LOG_INFO);
  logging::info("logging configured");

  if (optind < argc)
  {
    logging::error("Unrecognized argument:");
    while (optind < argc)
      logging::error("  %s ", argv[optind++]);
    usage(argv[0]);
    return 1;
  }

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

  // The number of event buffers cannot be greater than the system's max allowed.
  // See /proc/sys/fs/mqueue/msg_max for current value.  Use sysctl to change it.
  if (prms.numEvBuffers < NUMBEROF_XFERBUFFERS)
    prms.numEvBuffers = NUMBEROF_XFERBUFFERS;
  if (prms.numEvBuffers < prms.nevqueues)
    prms.numEvBuffers = prms.nevqueues; // Require numEvBuffers / nevqueues > 0
  if (prms.numEvBuffers > ImmData::MaxIdx)
  {
    logging::critical("Number of event buffers > %u is not supported: got %u",
                      ImmData::MaxIdx, prms.numEvBuffers);
    return 1;
  }

  if (prms.tag.empty())  prms.tag = prms.instrument;
  logging::info("Partition Tag: '%s'", prms.tag.c_str());

  get_kwargs(kwargs_str, prms.kwargs);
  for (const auto& kwargs : prms.kwargs)
  {
    if (kwargs.first == "forceEnet")    continue;
    if (kwargs.first == "ep_fabric")    continue;
    if (kwargs.first == "ep_domain")    continue;
    if (kwargs.first == "ep_provider")  continue;
    if (kwargs.first == "eb_timeout")   continue; // EbAppBase
    logging::critical("Unrecognized kwarg '%s=%s'",
                      kwargs.first.c_str(), kwargs.second.c_str());
    return 1;
  }

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
