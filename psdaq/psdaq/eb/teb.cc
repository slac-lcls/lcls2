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
#include "psdaq/service/Fifo.hh"
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
#include <chrono>
#include <Python.h>

#include "rapidjson/document.h"

#define UNLIKELY(expr)  __builtin_expect(!!(expr), 0)
#define LIKELY(expr)    __builtin_expect(!!(expr), 1)

#ifndef POSIX_TIME_AT_EPICS_EPOCH
#define POSIX_TIME_AT_EPICS_EPOCH 631152000u
#endif

using namespace rapidjson;
using namespace XtcData;
using namespace Pds;
using namespace Pds::Trg;

using json     = nlohmann::json;
using logging  = psalg::SysLog;
using string_t = std::string;
using ms_t     = std::chrono::milliseconds;
using us_t     = std::chrono::microseconds;

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

    struct Batch
    {
      Batch(const EbDgram* dgram, uint64_t dsts_, unsigned idx_) :
        start(dgram), end(dgram), dsts(dsts_), idx(idx_) {};
      const EbDgram* start;
      const EbDgram* end;
      uint64_t       dsts;
      unsigned       idx;
    };

    class Teb : public EbAppBase
    {
    public:
      Teb(const EbParams& prms, const MetricExporter_t& exporter);
    public:
      int      resetCounters();
      int      startConnection(std::string& tebPort, std::string& mrqPort);
      int      connect();
      int      configure(Trigger* object, unsigned prescale);
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
      void     _monitor(ResultDgram* rdg);
      void     _tryPost(const EbDgram* dg, uint64_t dsts, unsigned immData);
      void     _post(const Batch& batch);
      uint64_t _receivers(unsigned rogs) const;
    private:
      std::vector<EbLfCltLink*>    _l3Links;
      EbLfServer                   _mrqTransport;
      std::vector<EbLfSvrLink*>    _mrqLinks;
      unsigned                     _eventIdx;
      BatchManager                 _batMan;
      Batch                        _batch;
      std::vector<Fifo<unsigned> > _monBufLists;
    private:
      //uint64_t                     _trimmed;
      Trigger*                     _trigger;
      unsigned                     _prescale;
      unsigned                     _iMeb;
      unsigned                     _rogReserved[MAX_MRQS];
    private:
      unsigned                     _wrtCounter;
      uint64_t                     _pidPrv;
    private:
      uint64_t                     _eventCount;
      uint64_t                     _trCount;
      uint64_t                     _splitCount;
      uint64_t                     _batchCount;
      uint64_t                     _writeCount;
      uint64_t                     _monitorCount;
      uint64_t                     _nMonCount;
      uint64_t                     _mebCount[MAX_MEBS];
      uint64_t                     _prescaleCount;
      int64_t                      _latency;
      int64_t                      _trgTime;
    private:
      const EbParams&              _prms;
      EbLfClient                   _l3Transport;
      const MetricExporter_t&      _exporter;
    };
  };
};


using namespace Pds::Eb;

Teb::Teb(const EbParams&         prms,
         const MetricExporter_t& exporter) :
  EbAppBase     (prms, exporter, "TEB", TEB_TMO_MS),
  _mrqTransport (prms.verbose, prms.kwargs),
  _batch        {nullptr, 0, 0},
  //_trimmed      (0),
  _trigger      (nullptr),
  _iMeb         (0),
  _rogReserved  {0, 0, 0, 0},
  _pidPrv       (0),
  _eventCount   (0),
  _trCount      (0),
  _splitCount   (0),
  _batchCount   (0),
  _writeCount   (0),
  _monitorCount (0),
  _nMonCount    (0),
  _mebCount     {0, 0, 0, 0},
  _prescaleCount(0),
  _latency      (0),
  _trgTime      (0),
  _prms         (prms),
  _l3Transport  (prms.verbose, prms.kwargs),
  _exporter     (exporter)
{
  std::map<std::string, std::string> labels{{"instrument", prms.instrument},
                                            {"partition", std::to_string(prms.partition)},
                                            {"detname", prms.alias},
                                            {"alias", prms.alias}};
  exporter->add("TEB_EvtCt",  labels, MetricType::Counter, [&](){ return _eventCount;            });
  exporter->add("TEB_TrCt",   labels, MetricType::Counter, [&](){ return _trCount;               });
  exporter->add("TEB_SpltCt", labels, MetricType::Counter, [&](){ return _splitCount;            });
  exporter->add("TEB_BatCt",  labels, MetricType::Counter, [&](){ return _batchCount;            }); // Outbound
  exporter->add("TEB_TxPdg",  labels, MetricType::Gauge,   [&](){ return _l3Transport.posting(); });
  exporter->add("TEB_WrtCt",  labels, MetricType::Counter, [&](){ return _writeCount;            });
  exporter->add("TEB_MonCt",  labels, MetricType::Counter, [&](){ return _monitorCount;          });
  exporter->add("TEB_nMonCt", labels, MetricType::Counter, [&](){ return _nMonCount;             });
  exporter->add("TEB_MebCt0", labels, MetricType::Counter, [&](){ return _mebCount[0];           });
  exporter->add("TEB_MebCt1", labels, MetricType::Counter, [&](){ return _mebCount[1];           });
  exporter->add("TEB_MebCt2", labels, MetricType::Counter, [&](){ return _mebCount[2];           });
  exporter->add("TEB_MebCt3", labels, MetricType::Counter, [&](){ return _mebCount[3];           });
  exporter->add("TEB_PsclCt", labels, MetricType::Counter, [&](){ return _prescaleCount;         });
  exporter->add("TEB_EvtLat", labels, MetricType::Gauge,   [&](){ return _latency;               });
  exporter->add("TEB_trg_dt", labels, MetricType::Gauge,   [&](){ return _trgTime;               });
}

int Teb::resetCounters()
{
  EbAppBase::resetCounters();

  //_trimmed       = 0;
  _eventCount    = 0;
  _trCount       = 0;
  _splitCount    = 0;
  _batchCount    = 0;
  _writeCount    = 0;
  //_monitorCount  = 0;  // Cleared in Configure to stay in sync with MEB
  _prescaleCount = 0;
  _latency       = 0;
  _trgTime       = 0;

  return 0;
}

void Teb::shutdown()
{
  _mrqTransport.shutdown();

  EbAppBase::shutdown();
}

void Teb::disconnect()
{
  for (auto link : _mrqLinks)  _mrqTransport.disconnect(link);
  _mrqLinks.clear();

  for (auto link : _l3Links)  _l3Transport.disconnect(link);
  _l3Links.clear();

  EbAppBase::disconnect();

  _monBufLists.clear();
}

void Teb::unconfigure()
{
  if (!_l3Links.empty())              // Avoid dumping again if already done
    _batMan.dump();
  _batMan.shutdown();

  EbAppBase::unconfigure();
}

int Teb::startConnection(std::string& tebPort,
                         std::string& mrqPort)
{
  int rc = EbAppBase::startConnection(_prms.ifAddr, tebPort, MAX_DRPS);
  if (rc)  return rc;

  rc = _mrqTransport.listen(_prms.ifAddr, mrqPort, MAX_MRQS);
  if (rc)
  {
    logging::error("%s:\n  Failed to initialize %s EbLfServer on %s:%s",
                   __PRETTY_FUNCTION__, "MRQ", _prms.ifAddr.c_str(), mrqPort.c_str());
    return rc;
  }

  return 0;
}

int Teb::connect()
{
  _l3Links .resize(_prms.addrs.size());
  _mrqLinks.resize(_prms.numMrqs);

  for (unsigned i = 0; i < _prms.numMrqs; ++i)
    _monBufLists.emplace_back(_prms.numMebEvBufs);

  std::map<std::string, std::string> labels{{"instrument", _prms.instrument},
                                            {"partition", std::to_string(_prms.partition)},
                                            {"detname", _prms.alias},
                                            {"alias", _prms.alias}};
  for (unsigned i = 0; i < _monBufLists.size(); ++i)
    _exporter->add("TEB_MBufCt" + std::to_string(i), labels, MetricType::Gauge, [&, i](){ return _monBufLists[i].count(); });

  // Make a guess at the size of the Input entries
  size_t inpSizeGuess = sizeof(EbDgram) + 2  * sizeof(uint32_t);

  int rc = EbAppBase::connect(_prms, inpSizeGuess);
  if (rc)  return rc;

  rc = linksConnect(_l3Transport, _l3Links, _prms.addrs, _prms.ports, "DRP");
  if (rc)  return rc;
  rc = linksConnect(_mrqTransport, _mrqLinks, "MRQ");
  if (rc)  return rc;

  // Make a guess at the size of the Result entries
  auto maxResultSizeGuess = sizeof(EbDgram) + 2 * sizeof(uint32_t);
  auto maxEntries         = _prms.maxEntries;
  auto numBatches         = _prms.numBuffers / maxEntries;
  if (numBatches * maxEntries != _prms.numBuffers)
  {
    logging::critical("%s:\n  maxEntries (%u) must divide evenly into numBuffers (%u)",
                      maxEntries, _prms.numBuffers);
    abort();
  }
  _batMan.initialize(maxResultSizeGuess, maxEntries, numBatches); // TEB always batches

  // This is the local Results batch region from which we'll post batches back to the DRPs
  void*  region  = _batMan.batchRegion();
  size_t regSize = _batMan.batchRegionSize();

  for (auto link : _l3Links)
  {
    rc = link->setupMr(region, regSize);
    if (rc)  return rc;
  }

  return 0;
}

int Teb::configure(Trigger* trigger,
                   unsigned prescale)
{
  _monitorCount = 0; // Cleared here to stay in sync with MEB
  _nMonCount    = 0;
  for (unsigned i = 0; i < MAX_MEBS; ++ i)
    _mebCount[i] = 0;

  _trigger    = trigger;                // The trigger object
  _prescale   = prescale - 1;           // Be zero based
  _wrtCounter = _prescale;              // Reset prescale counter

  int rc = EbAppBase::configure(_prms);
  if (rc)  return rc;

  // maxResultSize becomes known during Configure, so reinitialize BatchManager now
  auto maxResultSize = _trigger->size();
  auto maxEntries    = _prms.maxEntries;
  auto numBatches    = _prms.numBuffers / maxEntries;
  _batMan.initialize(maxResultSize, maxEntries, numBatches); // TEB always batches

  // This is the local Results batch region
  void*  region  = _batMan.batchRegion();
  size_t regSize = _batMan.batchRegionSize();

  rc = linksConfigure(_l3Links, _prms.id, region, regSize, "DRP");
  if (rc)  return rc;
  rc = linksConfigure(_mrqLinks, _prms.id, "MRQ");
  if (rc)  return rc;

  // Code added here involving the links must be coordinated with the other side

  // Each rogReserved entry holds the number of buffers to be reserved for
  // events that include contributions from (a) "slow" readout group(s) in order
  // to increase the chances that a buffer is available when such an event comes
  // along.  If rogReserved is >= the number of MEB buffers in circulation, the
  // MEB will receive only events having a slow RoG contribution.  Similarly, to
  // avoid giving emphasis to any slow RoG, set rogReserved to 0.  To prevent
  // one slow RoG from starving another, rogReserved is a sum over all RoGs of
  // the requirement for each RoG.
  for (unsigned iMeb = 0; iMeb < MAX_MEBS; ++iMeb)
  {
    _rogReserved[iMeb] = 0;

    unsigned rogs = _prms.rogs & ~(1 << _prms.partition); // Skip the common RoG
    while (rogs)
    {
      unsigned rog = __builtin_ffs(rogs) - 1;
      rogs &= ~(1 << rog);
      _rogReserved[iMeb] += _trigger->rogReserve(rog, iMeb, _prms.numMebEvBufs);
    }
  }

  rc = _trigger->initialize(bufferSizes(), maxResultSize);
  if (rc)
  {
    logging::error("%s:\n  Failed to initialize trigger", __PRETTY_FUNCTION__);
    return rc;
  }

  return 0;
}

void Teb::run()
{
  logging::info("TEB thread started");

  int rc = pinThread(pthread_self(), _prms.core[0]);
  if (rc && _prms.verbose)
  {
    logging::error("%s:\n  Error pinning thread to core %d:\n  %m",
                   __PRETTY_FUNCTION__, _prms.core[0]);
  }

  _batch.start = nullptr;
  _batch.end   = nullptr;
  _batch.dsts  = 0;
  _batch.idx   = 0;
  _eventIdx    = 0;

  for (auto monBufList : _monBufLists)
    monBufList.clear();

  resetCounters();

  int rcPrv = 0;
  while (lRunning)
  {
    rc = EbAppBase::process();
    if (rc < 0)
    {
      if (rc == -FI_ENOTCONN)
      {
        logging::critical("TEB thread lost connection with a DRP");
        throw "Receiver thread lost connection with a DRP";
      }
      if (rc == rcPrv)  throw "Repeating fatal error";
    }
    rcPrv = rc;
  }

  logging::info("TEB thread finished");
}

void Teb::_monitor(ResultDgram* rdg)
{
  uint64_t immData;
  while (_mrqTransport.poll(&immData) > 0)
  {
    _monBufLists[ImmData::src(immData)].push(unsigned(immData));
  }

  auto allMebs{(1u << _prms.numMrqs) - 1};
  auto dsts{rdg->monitor() & allMebs};
  const bool roundRobin{dsts == allMebs};
  const auto numMebs{_monBufLists.size()};
  while (dsts)
  {
    unsigned iMeb;
    if (roundRobin)
    {
      iMeb = _iMeb;
      _iMeb = (iMeb + 1) % numMebs;
    }
    else
      iMeb = __builtin_ffs(dsts) - 1;

    dsts &= ~(1 << iMeb);

    if (_monBufLists[iMeb].count() > _rogReserved[iMeb])
    {
      unsigned buffer;
      _monBufLists[iMeb].pop(buffer);

      rdg->monBufNo(buffer);

      ++_monitorCount;
      ++_mebCount[iMeb];

      return;
    }
  }

  ++_nMonCount;                         // Count requests not monitored

  rdg->monitor(0);                      // Override monitor flags
}

void Teb::process(EbEvent* event)
{
  // Accumulate output datagrams (result) from the event builder into a batch
  // datagram.  When the batch completes, post it to the contributors.

  if (UNLIKELY(_prms.verbose >= VL_DETAILED))
  {
    printf("Teb::process event dump:\n");
    event->dump(_eventIdx);
  }

  const EbDgram* dgram = event->creator();
  if (!(dgram->readoutGroups() & (1 << _prms.partition)))
  {
    // The common readout group keeps events and batches in pulse ID order
    logging::error("%s:\n  Event %014lx, env %08x is missing the common readout group %u",
                   __PRETTY_FUNCTION__, dgram->pulseId(), dgram->env, _prms.partition);
    // Revisit: Should this be fatal?
  }

  unsigned imm = event->immData();
  uint64_t pid = dgram->pulseId();
  if (UNLIKELY(!(pid > _pidPrv)))
  {
    event->damage(Damage::OutOfOrder);

    logging::critical("%s:\n  Pulse ID did not advance: %014lx <= %014lx, rem %08lx, imm %08x, svc %u, ts %u.%09u\n",
                      __PRETTY_FUNCTION__, pid, _pidPrv, event->remaining(), imm, dgram->service(), dgram->time.seconds(), dgram->time.nanoseconds());

    if (event->remaining())             // I.e., this event was fixed up
    {
      // This can happen only for a split event (I think), which was fixed up and
      // posted earlier, so return to dismiss this counterpart and not post it
      // However, we can't know whether this is a split event or a fixed-up out-of-order event
      ++_splitCount;
      logging::critical("%s:\n  Split event, if pid %014lx was fixed up multiple times\n",
                        __PRETTY_FUNCTION__, pid);
      // return, if we could know this PID had been fixed up before
    }
    throw "Pulse ID did not advance";   // Can't recover from non-spit events
  }
  _pidPrv = pid;

  if (dgram->isEvent())  ++_eventCount;
  else                   ++_trCount;

  auto idx = _eventIdx++ & (_prms.numBuffers - 1);

  // "Selected" EBs respond with a Result, others simply acknowledge
  if (ImmData::rsp(ImmData::flg(imm)) == ImmData::Response)
  {
    auto buf = _batMan.fetch(idx);
    auto rdg = new(buf) ResultDgram(*dgram, _prms.id);

    rdg->xtc.damage.increase(event->damage().value());

    if (rdg->isEvent())
    {
      // Present event contributions to "user" code for building a result datagram
      auto t0 = std::chrono::system_clock::now();
      _trigger->event(event->begin(), event->end(), *rdg); // Consume
      auto t1 = std::chrono::system_clock::now();
      _trgTime = std::chrono::duration_cast<ns_t>(t1 - t0).count();

      // Handle prescale
      if (!rdg->persist() && !_wrtCounter--)
      {
        _prescaleCount++;

        rdg->prescale(true);
        _wrtCounter = _prescale;
      }

      if (rdg->persist())  _writeCount++;
      if (rdg->monitor())  _monitor(rdg);
    }

    // Avoid sending Results to contributors that failed to supply Input
    uint64_t dsts = _receivers(dgram->readoutGroups()) & ~event->remaining();

    if (UNLIKELY(_prms.verbose >= VL_EVENT)) // || rdg->monitor()))
    {
      const char* svc = TransitionId::name(rdg->service());
      uint64_t    pid = rdg->pulseId();
      unsigned    ctl = rdg->control();
      size_t      sz  = sizeof(rdg) + rdg->xtc.sizeofPayload();
      unsigned    src = rdg->xtc.src.value();
      unsigned    env = rdg->env;
      uint32_t*   pld = reinterpret_cast<uint32_t*>(rdg->xtc.payload());
      printf("TEB processed %15s result [%8u] @ "
             "%16p, ctl %02x, pid %014lx, env %08x, sz %6zd, src %2u, dsts %016lx, res [%08x, %08x]\n",
             svc, idx, rdg, ctl, pid, env, sz, src, dsts, pld[0], pld[1]);
    }

    _tryPost(rdg, dsts, idx);
  }
  else                                  // "Non-selected" TEB case
  {
    // Only transitions are sent to "non-selected" TEBs.
    // "Non-selected" TEBs don't respond to any dgrams they receive, but
    // responses prepared for dgrams they received when they were a "selected"
    // TEB must be delivered in a timely manner, so whatever in-progress batch
    // there is is flushed by the same logic batches on "selected" TEBs are
    // flushed.  It's probably done by the first SlowUpdate after a TEB becomes
    // a "non-selected" one, so this seems a bit redundant.
    if (_batch.start)
    {
      TransitionId::Value svc     = dgram->service();
      bool                flush   = !((svc == TransitionId::L1Accept) ||
                                      (svc == TransitionId::SlowUpdate));
      bool                expired = _batMan.expired(pid, _batch.start->pulseId());

      if (expired || flush)
      {
        if (!_batch.end)  _batch.end = _batch.start;

        _post(_batch);                  // Flush whatever batch there is

        _batch.start = nullptr;         // Start a new batch
      }
    }

    if (UNLIKELY(_prms.verbose >= VL_EVENT)) // || rdg->monitor()))
    {
      const char* svc = TransitionId::name(dgram->service());
      unsigned    ctl = dgram->control();
      size_t      sz  = sizeof(dgram) + dgram->xtc.sizeofPayload();
      unsigned    src = dgram->xtc.src.value();
      unsigned    env = dgram->env;
      printf("TEB processed %15s ACK    [%8u] @ "
             "%16p, ctl %02x, pid %014lx, env %08x, sz %6zd, src %2u, data %08x\n",
             svc, idx, dgram, ctl, pid, env, sz, src, imm);
    }

    // Make the transition buffer available to the contributor again
    post(event->begin(), event->end());
  }

  auto now = std::chrono::system_clock::now();
  auto dgt = std::chrono::seconds{dgram->time.seconds() + POSIX_TIME_AT_EPICS_EPOCH}
           + std::chrono::nanoseconds{dgram->time.nanoseconds()};
  std::chrono::system_clock::time_point tp{std::chrono::duration_cast<std::chrono::system_clock::duration>(dgt)};
  _latency = std::chrono::duration_cast<ms_t>(now - tp).count();
}

// Called by EB  on timeout when it is empty of events
// to flush out any in-progress batch
void Teb::flush()
{
  //printf("TEB::flush: start %p, end %p\n", _batch.start, _batch.end);

  if (_batch.start)
  {
    //printf("TEB::flush:    posting %014lx - %014lx\n",
    //       _batch.start->pulseId(), _batch.end->pulseId());

    _post(_batch);

    _batch.start = nullptr;             // Start a new batch
  }
}

void Teb::_tryPost(const EbDgram* dgram, uint64_t dsts, unsigned eventIdx)
{
  // On wrapping, post the batch at the end of the region, if any
  if (dgram == _batMan.batchRegion())  flush();

  // The batch start is the first dgram seen
  if (!_batch.start)  _batch = {dgram, dsts, eventIdx};

  TransitionId::Value svc     = dgram->service();
  bool                flush   = ((svc != TransitionId::L1Accept) &&
                                 (svc != TransitionId::SlowUpdate));
  bool                expired = _batMan.expired(       dgram->pulseId(),
                                                _batch.start->pulseId());

  if (LIKELY(!expired && !flush))       // Most frequent case when batching
  {
    _batch.end   = dgram;               // Append dgram to batch
    _batch.dsts |= dsts;
  }
  else
  {
    // Combining a flushing dgram (i.e., a non-SlowUpdate transition) into an
    // expired batch can lead to downstream problems since the transition's
    // pulseId may fall outside the batch duration (epoch)
    if (expired)                        // Post just the batch
    {
      _post(_batch);                    // The batch end is the previous Dgram

      _batch = {dgram, dsts, eventIdx}; // Start a new batch with dgram
    }

    if (flush)                          // Post the batch + transition
    {
      _batch.end   = dgram;             // Append dgram to batch
      _batch.dsts |= dsts;

      _post(_batch);

      _batch.start = nullptr;           // Start a new batch
    }
  }
}

void Teb::_post(const Batch& batch)
{
  size_t   size   = _trigger->size();
  size_t   extent = (reinterpret_cast<const char*>(batch.end) -
                     reinterpret_cast<const char*>(batch.start)) + size;
  unsigned offset = batch.idx * size;
  uint64_t data   = ImmData::value(ImmData::Buffer, _prms.id, batch.idx);
  uint64_t destns = batch.dsts; // & ~_trimmed;

  batch.end->setEOL();                  // Terminate the batch

  if (UNLIKELY(_prms.verbose >= VL_BATCH))
  {
    uint64_t pid = batch.start->pulseId();
    printf("TEB posts          %9lu result  [%8u] @ "
           "%16p,         pid %014lx, ofs %08x, sz %6zd, dst %016lx\n",
           _batchCount, batch.idx, batch.start, pid, offset, extent, destns);
  }

  while (destns)
  {
    unsigned     dst  = __builtin_ffsl(destns) - 1;
    EbLfCltLink* link = _l3Links[dst];

    destns &= ~(1ul << dst);

    if (UNLIKELY(_prms.verbose >= VL_BATCH))
    {
      void* rmtAdx = (void*)link->rmtAdx(offset);
      printf("                                      to DRP %2u @ %16p\n",
             dst, rmtAdx);
    }

    int rc = link->post(batch.start, extent, offset, data);
    if (rc < 0)
    {
      uint64_t pid    = batch.start->pulseId();
      unsigned ctl    = batch.start->control();
      uint32_t env    = batch.start->env;
      void*    rmtAdx = (void*)link->rmtAdx(offset);
      logging::critical("%s:\n  Failed to post batch  [%8u]  @ "
                        "%16p, ctl %02x, pid %014lx, env %08x, sz %6zd, TEB %2u @ %16p, data %08x, rc %d\n",
                        __PRETTY_FUNCTION__, batch.idx, batch.start, ctl, pid, env, extent, dst, rmtAdx, data, rc);
      abort();

      // If we were to trim, here's how to do it.  For now, we don't.
      //static unsigned retries = 0;
      //trim(dst);
      //if (retries++ == 5)  { _trimmed |= 1ul << dst; retries = 0; }
      //printf("%s:  link->post() to %u returned %d, trimmed = %016lx\n",
      //       __PRETTY_FUNCTION__, dst, rc, _trimmed);
    }
  }

  ++_batchCount;
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

    receivers |= _prms.receivers[group];
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
  virtual ~TebApp();
public:                                 // For CollectionApp
  json connectionInfo() override;
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
  void _printParams(const EbParams& prms, Trigger* trigger) const;
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

void TebApp::connectionShutdown()
{
  _teb->shutdown();
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
    _error(msg, "Connection parameters error - see log");
    return;
  }

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

  std::string buildDets("---");
  if (top.HasMember("buildDets"))
    buildDets = top["buildDets"].GetString();

  for (auto it : body["drp"].items())
  {
    unsigned    drpId   = it.value()["drp_id"];
    std::string alias   = it.value()["proc_info"]["alias"];
    size_t      found   = alias.rfind('_');
    std::string detName = alias.substr(0, found);

    if (buildAll || buildDets.find(detName))
    {
      unsigned group(it.value()["det_info"]["readout"]);
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

  if (trigger->configure(_connectMsg, top, _prms))
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

  _printParams(_prms, trigger);

  _teb->resetCounters();                // Same time as DRPs

  return rc;
}

void TebApp::_unconfigure()
{
  // Shut down the previously running instance, if any
  lRunning = 0;
  if (_appThread.joinable())  _appThread.join();

  _teb->unconfigure();

  _unconfigFlag = false;
}

void TebApp::handlePhase1(const json& msg)
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

    _appThread = std::thread(&Teb::run, std::ref(*_teb));
  }
  else if (key == "unconfigure")
  {
    // "Queue" unconfiguration until after phase 2 has completed
    _unconfigFlag = true;
  }
  else if (key == "beginrun")
  {
    _teb->resetCounters();              // Same time as DRPs
  }

  // Reply to collection with transition status
  reply(createMsg(key, msg["header"]["msg_id"], getId(), body));
}

void TebApp::handleDisconnect(const json& msg)
{
  // Carry out the queued Unconfigure, if there was one
  if (_unconfigFlag)  _unconfigure();

  _teb->disconnect();

  // Reply to collection with transition status
  json body = json({});
  reply(createMsg("disconnect", msg["header"]["msg_id"], getId(), body));
}

void TebApp::handleReset(const json& msg)
{
  unsubscribePartition();               // ZMQ_UNSUBSCRIBE

  _unconfigure();
  _teb->disconnect();
  connectionShutdown();
}

int TebApp::_parseConnectionParams(const json& body)
{
  int rc = 0;

  std::string id = std::to_string(getId());
  _prms.id = body["teb"][id]["teb_id"];
  if (_prms.id >= MAX_TEBS)
  {
    logging::error("TEB ID %d is out of range 0 - %u", _prms.id, MAX_TEBS - 1);
    rc = 1;
  }

  if (body.find("drp") == body.end())
  {
    logging::error("Missing required DRP specs");
    rc = 1;
  }

  _prms.contributors = 0;
  _prms.addrs.clear();
  _prms.ports.clear();

  _prms.rogs = 0;
  _prms.contractors.fill(0);
  _prms.receivers.fill(0);

  _prms.maxEntries  = MAX_ENTRIES;      // Revisit: Make configurable?
  _prms.numBuffers  = 0;                // Only need the largest value

  unsigned maxBuffers = 0;
  for (auto it : body["drp"].items())
  {
    unsigned drpId = it.value()["drp_id"];
    if (drpId > MAX_DRPS - 1)
    {
      logging::error("DRP ID %d is out of range 0 - %u", drpId, MAX_DRPS - 1);
      rc = 1;
    }
    _prms.contributors |= 1ul << drpId;

    _prms.addrs.push_back(it.value()["connect_info"]["nic_ip"]);
    _prms.ports.push_back(it.value()["connect_info"]["drp_port"]);

    unsigned rog(it.value()["det_info"]["readout"]);
    if (rog > NUM_READOUT_GROUPS - 1)
    {
      logging::error("Readout group %u is out of range 0 - %u", rog, NUM_READOUT_GROUPS - 1);
      rc = 1;
    }
    _prms.rogs             |= 1 << rog;
    _prms.contractors[rog] |= 1ul << drpId; // Possibly overridden during Configure
    _prms.receivers[rog]   |= 1ul << drpId; // All contributors receive results

    // The Common RoG governs the index into the Results region.
    // Its range must be >= that of any subsidiary RoG.
    auto numBuffers = it.value()["connect_info"]["num_buffers"];
    if (numBuffers > _prms.numBuffers)
    {
      if (rog == _prms.partition)
        _prms.numBuffers = numBuffers;
      else if (numBuffers > maxBuffers)
        maxBuffers = numBuffers;
    }
  }

  if (_prms.numBuffers & (_prms.numBuffers - 1))
  {
    logging::error("numBuffers (%u = 0x%08x) isn't a power of 2, as it should be",
                   _prms.numBuffers, _prms.numBuffers);
    rc = 1;
  }

  // Disallow non-common RoG DRPs from having more buffers than the common one
  // because the buffer index based on the common RoG DRPs won't be able to
  // reach the higher buffer numbers.  Can't use an index based on the largest
  // non-common RoG DRP because it would overrun the common RoG DRPs' region.
  if (maxBuffers > _prms.numBuffers)
  {
    logging::error("DRP's DMA buffer count (%u) must be <= %u",
                   maxBuffers, _prms.numBuffers);
    rc = 1;
  }

  // These buffers aren't used if there is only one TEB in the system, but...
  unsigned suRate(body["control"]["0"]["control_info"]["slow_update_rate"]);
  if (1000 * TEB_TR_BUFFERS < suRate * TEB_TMO_MS)
  {
    // Adjust TEB_TMO_MS, TEB_TR_BUFFERS (in eb.hh) or the SlowUpdate rate
    logging::error("Increase # of TEB transition buffers from %u to > %u "
                   "for %u Hz of SlowUpdates and %u ms TEB timeout",
                   TEB_TR_BUFFERS, (suRate * TEB_TMO_MS + 999) / 1000, suRate, TEB_TMO_MS);
    rc = 1;
  }

  auto& vec =_prms.maxTrSize;
  vec.resize(body["drp"].size());
  std::fill(vec.begin(), vec.end(), sizeof(EbDgram)); // Same for all contributors

  _prms.numMrqs      = 0;
  _prms.numMebEvBufs = 0;
  if (body.find("meb") != body.end())
  {
    // Revisit: For now, we follow the DRP pattern, but we shouldn't require
    //          the number of MEB event buffers to be the same across MEBs
    for (auto it : body["meb"].items())
    {
      _prms.numMrqs++;    // Revisit: body.count("meb"); doesn't work?

      unsigned mebId = it.value()["meb_id"];
      unsigned count = it.value()["connect_info"]["max_ev_count"];
      if (_prms.numMebEvBufs == 0)
        _prms.numMebEvBufs = count;
      else if (count != _prms.numMebEvBufs)
      {
        logging::error("numMebEvBufs (%u) must be the same for all MEBs, got %u from ID %u",
                       _prms.numMebEvBufs, count, mebId);
        rc = 1;
      }
    }
    if (_prms.numMrqs > MAX_MRQS)
    {
      logging::error("More monitor requestors found (%u) than supportable %u",
                     _prms.numMrqs, MAX_MRQS);
      rc = 1;
    }
  }

  return rc;
}

static void _printGroups(unsigned groups, const EbAppBase::u64arr_t& array)
{
  while (groups)
  {
    unsigned group = __builtin_ffs(groups) - 1;
    groups &= ~(1 << group);

    printf("%u: 0x%016lx  ", group, array[group]);
  }
  printf("\n");
}

void TebApp::_printParams(const EbParams& prms, Trigger* trigger) const
{
  printf("Parameters of TEB ID %d (%s:%s):\n",                   prms.id,
                                                                 prms.ifAddr.c_str(), prms.ebPort.c_str());
  printf("  Thread core numbers:          %d, %d\n",             prms.core[0], prms.core[1]);
  printf("  Partition:                    %u\n",                 prms.partition);
  printf("  Bit list of contributors:     0x%016lx, cnt: %zu\n", prms.contributors,
                                                                 std::bitset<64>(prms.contributors).count());
  printf("  Readout group contractors:    ");                    _printGroups(prms.rogs, prms.contractors);
  printf("  Readout group receivers:      ");                    _printGroups(prms.rogs, prms.receivers);
  printf("  Number of MEB requestors:     %u\n",                 prms.numMrqs);
  printf("  Batch duration:               0x%08x = %u ticks\n",  prms.maxEntries, prms.maxEntries);
  printf("  Batch pool depth:             0x%08x = %u\n",        prms.numBuffers / prms.maxEntries, prms.numBuffers / prms.maxEntries);
  printf("  Max # of entries / batch:     0x%08x = %u\n",        prms.maxEntries, prms.maxEntries);
  printf("  # of contrib. buffers:        0x%08x = %u\n",        prms.numBuffers, prms.numBuffers);
  printf("  Max result     EbDgram size:  0x%08zx = %zu\n",      trigger->size(), trigger->size());
  printf("  Max transition EbDgram size:  0x%08zx = %zu\n",      prms.maxTrSize[0], prms.maxTrSize[0]);
  printf("  # of transition buffers:      0x%08x = %u\n",        TEB_TR_BUFFERS, TEB_TR_BUFFERS);
  printf("\n");
}


static
void usage(const char *name, const char *desc, const EbParams& prms)
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
  prms.partition  = NO_PARTITION;
  prms.core[0]    = CORE_0;
  prms.core[1]    = CORE_1;
  prms.verbose    = 0;

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
        usage(argv[0], "Trigger Event Builder application", prms);
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
    usage(argv[0], "Trigger Event Builder application", prms);
    return 1;
  }

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
  for (const auto& kwargs : prms.kwargs)
  {
    if (kwargs.first == "forceEnet")    continue;
    if (kwargs.first == "ep_fabric")    continue;
    if (kwargs.first == "ep_domain")    continue;
    if (kwargs.first == "ep_provider")  continue;
    if (kwargs.first == "script_path")  continue;
    logging::critical("Unrecognized kwarg '%s=%s'\n",
                      kwargs.first.c_str(), kwargs.second.c_str());
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
