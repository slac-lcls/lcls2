#include "TebContributor.hh"

#include "Endpoint.hh"
#include "EbLfClient.hh"
#include "EbCtrbInBase.hh"

#include "utilities.hh"

#include "psalg/utils/SysLog.hh"
#include "psdaq/service/MetricExporter.hh"
#include "xtcdata/xtc/Dgram.hh"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <string.h>
#include <cassert>
#include <cstdint>
#include <bitset>
#include <string>
#include <thread>
#include <chrono>

#include <unistd.h>

#define UNLIKELY(expr)  __builtin_expect(!!(expr), 0)
#define LIKELY(expr)    __builtin_expect(!!(expr), 1)

#ifndef POSIX_TIME_AT_EPICS_EPOCH
#define POSIX_TIME_AT_EPICS_EPOCH 631152000u
#endif

using namespace XtcData;
using namespace Pds;
using namespace Pds::Eb;
using logging = psalg::SysLog;
using ms_t    = std::chrono::milliseconds;


TebContributor::TebContributor(const TebCtrbParams&                   prms,
                               unsigned                               numBuffers,
                               const std::shared_ptr<MetricExporter>& exporter) :
  _prms       (prms),
  _transport  (prms.verbose, prms.kwargs),
  _id         (-1),
  _numEbs     (0),
  _pending    (numBuffers),
  _batch      {nullptr, false},
  _previousPid(0),
  _eventCount (0),
  _batchCount (0),
  _latency    (0)
{
  std::map<std::string, std::string> labels{{"instrument", prms.instrument},
                                            {"partition", std::to_string(prms.partition)},
                                            {"detname", prms.detName},
                                            {"detseg", std::to_string(prms.detSegment)},
                                            {"alias", prms.alias}};

  exporter->constant("TCtb_IUMax",  labels, MAX_BATCHES);
  exporter->constant("TCtbO_IFMax", labels, _pending.size());

  exporter->add("TCtbO_EvtCt", labels, MetricType::Counter, [&](){ return _eventCount;          });
  exporter->add("TCtbO_BatCt", labels, MetricType::Counter, [&](){ return _batchCount;          });
  exporter->add("TCtbO_TxPdg", labels, MetricType::Gauge,   [&](){ return _transport.posting(); });
  exporter->add("TCtbO_InFlt", labels, MetricType::Gauge,   [&](){ _pendingSize = _pending.guess_size();
                                                                   return _pendingSize; });
  exporter->add("TCtbO_Lat",   labels, MetricType::Gauge,   [&](){ return _latency;             });
}

TebContributor::~TebContributor()
{
  // Try to take things down gracefully when an exception takes us off the
  // normal path so that the most chance is given for prints to show up
  shutdown();
}

int TebContributor::resetCounters()
{
  _eventCount = 0;
  _batchCount = 0;

  return 0;
}

void TebContributor::startup(EbCtrbInBase& in)
{
  _batch.start = nullptr;
  _batch.end   = nullptr;

  resetCounters();
  in.resetCounters();

  _running.store(true, std::memory_order_release);
  _rcvrThread = std::thread([&] { in.receiver(*this, _running); });
}

void TebContributor::shutdown()
{
  if (!_links.empty())                  // Avoid shutting down if already done
  {
    unconfigure();
    disconnect();
  }
}

void TebContributor::disconnect()
{
  for (auto link : _links)  _transport.disconnect(link);
  _links.clear();

  _id = -1;
}

void TebContributor::unconfigure()
{
  if (!_links.empty())             // Avoid unconfiguring again if already done
  {
    _running.store(false, std::memory_order_release);

    if (_rcvrThread.joinable())  _rcvrThread.join();

    _batMan.dump();
    _batMan.shutdown();
    _pending.shutdown();
  }
}

int TebContributor::connect(size_t inpSizeGuess)
{
  _links    .resize(_prms.addrs.size());
  _trBuffers.resize(_links.size());
  _id       = _prms.id;
  _numEbs   = std::bitset<64>(_prms.builders).count();

  int rc = linksConnect(_transport, _links, _prms.addrs, _prms.ports, _id, "TEB");
  if (rc)  return rc;

  // Set up a guess at the RDMA region
  // If it's too small, it will be corrected during Configure
  if (!_batMan.batchRegion())           // No need to guess again
  {
    auto numBatches = _pending.size() / _prms.maxEntries;
    if (numBatches * _prms.maxEntries != _pending.size())
    {
      logging::critical("%s:\n  maxEntries (%u) must divide evenly into numBuffers (%u)",
                        _prms.maxEntries, _pending.size());
      abort();
    }
    _batMan.initialize(inpSizeGuess, _prms.maxEntries, numBatches);
  }

  void*  region  = _batMan.batchRegion();     // Local space for Trs is in the batch region
  size_t regSize = _batMan.batchRegionSize(); // No need to add Tr space size here

  for (auto link : _links)
  {
    printf("ID %2u: ", link->id());
    rc = link->setupMr(region, regSize);
    if (rc)  return rc;
  }

  return 0;
}

int TebContributor::configure()
{
  // To give maximal chance of inspection with a debugger of a previous run's
  // information, clear it in configure() rather than in unconfigure()
  const EbDgram* dg;
  while (_pending.try_pop(dg));
  _pending.startup();

  // maxInputSize becomes known during Configure, so reinitialize BatchManager now
  auto numBatches = _pending.size() / _prms.maxEntries;
  _batMan.initialize(_prms.maxInputSize, _prms.maxEntries, numBatches);

  void*  region  = _batMan.batchRegion();     // Local space for Trs is in the batch region
  size_t regSize = _batMan.batchRegionSize(); // No need to add Tr space size here

  int rc = linksConfigure(_links, region, regSize, _prms.maxInputSize, "TEB");
  if (rc)  return rc;

  // Code added here involving the links must be coordinated with the other side

  for (auto link : _links)
  {
    auto& lst = _trBuffers[link->id()];
    lst.clear();

    for (unsigned buf = 0; buf < TEB_TR_BUFFERS; ++buf)
    {
      lst.push_back(buf);
    }
  }

  return 0;
}

void TebContributor::_flush()
{
  if (_batch.start)
  {
    _post(_batch);
    _batch.start = nullptr;             // Start a new batch
  }
}

// NB: timeout() must not be called concurrently with process()
void TebContributor::timeout()
{
  _flush();
}

// NB: process() must not be called concurrently with timeout()
void TebContributor::process(const EbDgram* dgram)
{
  auto now = std::chrono::system_clock::now();
  auto dgt = std::chrono::seconds{dgram->time.seconds() + POSIX_TIME_AT_EPICS_EPOCH}
           + std::chrono::nanoseconds{dgram->time.nanoseconds()};
  std::chrono::system_clock::time_point tp{std::chrono::duration_cast<std::chrono::system_clock::duration>(dgt)};
  _latency = std::chrono::duration_cast<ms_t>(now - tp).count();

  auto rogs       = dgram->readoutGroups();
  bool contractor = rogs & _prms.contractor; // T if providing TEB input

  if (LIKELY(rogs & (1 << _prms.partition))) // Common RoG triggered
  {
    // On wrapping, post the batch at the end of the region, if any
    if (dgram == _batMan.batchRegion())  _flush();

    // Initiailize a new batch with the first dgram seen
    if (!_batch.start)  _batch = {dgram, contractor};

    auto svc     = dgram->service();
    bool doFlush = ((svc != TransitionId::L1Accept) &&
                    (svc != TransitionId::SlowUpdate));
    bool expired = _batMan.expired(       dgram->pulseId(),
                                   _batch.start->pulseId());

    if (LIKELY(!expired && !doFlush))   // Most frequent case when batching
    {
      _batch.end         = dgram;       // Append dgram to batch
      _batch.contractor |= contractor;
    }
    else
    {
      // Combining a flushing dgram (i.e., a non-SlowUpdate transition) into an
      // expired batch can lead to downstream problems since the transition's
      // pulseId may fall outside the batch duration (epoch)
      if (expired)                      // Post just the batch
      {
        _post(_batch);                  // The batch end is the previous Dgram

        _batch = {dgram, contractor};   // Start a new batch with dgram
      }

      if (doFlush)                      // Post the batch + transition
      {
        _batch.end         = dgram;     // Append dgram to batch
        _batch.contractor |= contractor;

        _post(_batch);

        _batch.start = nullptr;         // Start a new batch
      }
    }

    // Keep non-selected TEBs synchronized by forwarding transitions to them.  In
    // particular, the Disable transition flushes out whatever Results batch they
    // currently have in-progress.
    if (!dgram->isEvent())           // Also capture the most recent SlowUpdate
    {
      if (contractor)  _post(dgram); // Post, if contributor is providing trigger input
    }
  }
  else                        // Common RoG didn't trigger: bypass the TEB(s)
  {
    _flush();

    dgram->setEOL();          // Terminate for clarity and dump-ability
    _pending.push(dgram);
    if (!(size_t(_pending.guess_size()) < _pending.size()))
    {
      logging::critical("%s: _pending queue overflow", __PRETTY_FUNCTION__);
      abort();
    }
  }

  ++_eventCount;                        // Only count events handled
}

void TebContributor::_post(const Batch& batch)
{
  batch.end->setEOL();        // Avoid race: terminate before adding batch to pending list
  _pending.push(batch.start); // Get the batch on the queue before any corresponding result can show up
  if (!(size_t(_pending.guess_size()) < _pending.size()))
  {
    logging::critical("%s: _pending queue overflow", __PRETTY_FUNCTION__);
    abort();
  }

  if (batch.contractor) // Send to TEB if contributor is providing trigger input
  {
    uint64_t     pid    = batch.start->pulseId();
    unsigned     dst    = (pid / _prms.maxEntries) % _numEbs;
    EbLfCltLink* link   = _links[dst];
    unsigned     offset = link->lclOfs(batch.start);
    uint32_t     idx    = offset / _prms.maxInputSize;
    size_t       extent = (reinterpret_cast<const char*>(batch.end) -
                           reinterpret_cast<const char*>(batch.start)) + _prms.maxInputSize;
    uint32_t     data   = ImmData::value(ImmData::Buffer |
                                         ImmData::Response, _id, idx);

    if (UNLIKELY(_prms.verbose >= VL_BATCH))
    {
      void* rmtAdx = (void*)link->rmtAdx(offset);
      printf("CtrbOut posts %9lu    batch[%8u]    @ "
             "%16p,         pid %014lx,               sz %6zd, TEB %2u @ %16p, data %08x\n",
             _batchCount, idx, batch.start, pid, extent, dst, rmtAdx, data);
    }
    else
    {
      auto dgram = batch.start;
      auto svc   = dgram->service();
      if (svc != XtcData::TransitionId::L1Accept) {
        void* rmtAdx = (void*)link->rmtAdx(offset);
        if (svc != XtcData::TransitionId::SlowUpdate) {
          logging::info("TebCtrb   sent %s @ %u.%09u (%014lx) to TEB ID %u @ %16p (%08x + %u * %08zx)",
                        XtcData::TransitionId::name(svc),
                        dgram->time.seconds(), dgram->time.nanoseconds(),
                        dgram->pulseId(), dst, rmtAdx, 0, idx, _prms.maxInputSize);
        }
        else {
          logging::debug("TebCtrb   sent %s @ %u.%09u (%014lx) to TEB ID %u @ %16p (%08x + %u * %08zx)",
                         XtcData::TransitionId::name(svc),
                         dgram->time.seconds(), dgram->time.nanoseconds(),
                         dgram->pulseId(), dst, rmtAdx, 0, idx, _prms.maxInputSize);
        }
      }
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
                        __PRETTY_FUNCTION__, idx, batch.start, ctl, pid, env, extent, dst, rmtAdx, data, rc);
      abort();
    }
  }

  ++_batchCount;                        // Count all batches handled
}

// This is the same as in MebContributor as we have no good common place for it
static int _getTrBufIdx(EbLfLink* lnk, TebContributor::listU32_t& lst, uint32_t& idx)
{
  // Try to replenish the transition buffer index list
  while (true)
  {
    uint64_t imm;
    int rc = lnk->poll(&imm);           // Attempt to get a free buffer index
    if (rc)  break;
    lst.push_back(ImmData::idx(imm));
  }

  // If the list is still empty, wait for one
  if (lst.empty())
  {
    uint64_t imm;
    unsigned tmo = 5000;
    int rc = lnk->poll(&imm, tmo);      // Wait for a free buffer index
    if (rc)  return rc;
    idx = ImmData::idx(imm);
    return 0;
  }

  // Return the index at the head of the list
  idx = lst.front();
  lst.pop_front();

  return 0;
}

void TebContributor::_post(const EbDgram* dgram)
{
  // Send transition datagrams to all TEBs, except the one that got the
  // batch containing it.  These TEBs don't generate responses.
  if (_links.size() < 2)  return;

  // Modifying dgram interferes with batch posted above: see comment in EbAppBase
  //dgram->setEOL();                      // Terminate the "batch" of 1 entry

  uint64_t pid = dgram->pulseId();
  unsigned dst = (pid / _prms.maxEntries) % _numEbs;
  size_t   sz  = sizeof(*dgram) + dgram->xtc.sizeofPayload();

  if (sz > sizeof(*dgram))
  {
    auto svc = dgram->service();
    logging::critical("%s transition has unexpected XTC payload of size %zd",
                      TransitionId::name(svc), dgram->xtc.sizeofPayload());
    abort();
  }

  for (auto link : _links)
  {
    unsigned src = link->id();
    if (src != dst)      // Skip dst, which received batch including this Dgram
    {
      uint32_t idx;
      int rc = _getTrBufIdx(link, _trBuffers[src], idx);
      if (rc)
      {
        auto svc = dgram->service();
        auto ts  = dgram->time;
        logging::critical("%s:\n  No transition buffer index received from TEB ID %u "
                          "needed for %s (%014lx, %9u.%09u): rc %d",
                          __PRETTY_FUNCTION__, src, TransitionId::name(svc), pid, ts.seconds(), ts.nanoseconds(), rc);
        abort();
      }

      unsigned offset = _batMan.batchRegionSize() + idx * sizeof(*dgram);
      uint32_t data   = ImmData::value(ImmData::Transition |
                                       ImmData::NoResponse, _id, idx);

      if (UNLIKELY(_prms.verbose >= VL_BATCH))
      {
        unsigned    env    = dgram->env;
        unsigned    ctl    = dgram->control();
        const char* svc    = TransitionId::name(dgram->service());
        void*       rmtAdx = (void*)link->rmtAdx(offset);
        printf("CtrbOut posts    %15s              @ "
               "%16p, ctl %02x, pid %014lx, env %08x, sz %6zd, TEB %2u @ %16p, data %08x\n",
               svc, dgram, ctl, pid, env, sz, src, rmtAdx, data);
      }
      else
      {
        auto svc = dgram->service();
        if (svc != XtcData::TransitionId::L1Accept) {
          void* rmtAdx = (void*)link->rmtAdx(offset);
          if (svc != XtcData::TransitionId::SlowUpdate) {
            logging::info("TebCtrb   sent %s @ %u.%09u (%014lx) to TEB ID %u @ %16p (%08zx + %u * %08zx)",
                          XtcData::TransitionId::name(svc),
                          dgram->time.seconds(), dgram->time.nanoseconds(),
                          dgram->pulseId(), src, rmtAdx, _batMan.batchRegionSize(), idx, sizeof(*dgram));
          }
          else {
            logging::debug("TebCtrb   sent %s @ %u.%09u (%014lx) to TEB ID %u @ %16p (%08zx + %u * %08zx)",
                           XtcData::TransitionId::name(svc),
                           dgram->time.seconds(), dgram->time.nanoseconds(),
                           dgram->pulseId(), src, rmtAdx, _batMan.batchRegionSize(), idx, sizeof(*dgram));
          }
        }
      }

      rc = link->post(dgram, sz, offset, data); // Not a batch; Continue on error
      if (rc)
      {
        logging::error("%s:\n  Failed to post buffer number to TEB ID %u: rc %d, data %08x",
                       __PRETTY_FUNCTION__, src, rc, data);
      }
    }
  }
}
