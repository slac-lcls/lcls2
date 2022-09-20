#include "EbCtrbInBase.hh"

#include "Endpoint.hh"
#include "EbLfServer.hh"
#include "TebContributor.hh"
#include "ResultDgram.hh"

#include "utilities.hh"

#include "psalg/utils/SysLog.hh"
#include "psdaq/service/MetricExporter.hh"
#include "psdaq/service/EbDgram.hh"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <string>
#include <bitset>
#include <chrono>

#define UNLIKELY(expr)  __builtin_expect(!!(expr), 0)
#define LIKELY(expr)    __builtin_expect(!!(expr), 1)


using namespace XtcData;
using namespace Pds;
using namespace Pds::Eb;
using logging  = psalg::SysLog;
using ms_t     = std::chrono::milliseconds;


static void dumpBatch(const TebContributor& ctrb,
                      const EbDgram*        batch,
                      const size_t          size,
                      unsigned              index)
{
  auto dg   = batch;
  auto bPid = dg->pulseId();
  for (unsigned i = 0; i < MAX_ENTRIES; ++i)
  {
    auto pid = dg->pulseId();
    auto svc = TransitionId::name(dg->service());
    auto rog = dg->readoutGroups();
    auto dmg = dg->xtc.damage.value();
    printf("  %2u: %16p, %15s, pid %014lx, diff %016lx, RoG %2hx, dmg %04x, idx %u %s\n",
           i, dg, svc, pid, pid - bPid, rog, dmg, index, dg->isEOL() ? "EOL" : "");
    if (dg->isEOL())  return;
    dg = reinterpret_cast<const EbDgram*>(reinterpret_cast<const char*>(dg) + size);
    ++index;
  }
  printf("  EOL not found!\n");
}


EbCtrbInBase::EbCtrbInBase(const TebCtrbParams&                   prms,
                           const std::shared_ptr<MetricExporter>& exporter) :
  _transport    (prms.verbose, prms.kwargs),
  _maxResultSize(0),
  _batchCount   (0),
  _eventCount   (0),
  _missing      (0),
  _bypassCount  (0),
  _noProgCount  (0),
  _prvNPCnt     (0),
  _prms         (prms),
  _regSize      (0),
  _region       (nullptr)
{
  std::map<std::string, std::string> labels{{"instrument", prms.instrument},
                                            {"partition", std::to_string(prms.partition)},
                                            {"detname", prms.detName},
                                            {"detseg", std::to_string(prms.detSegment)},
                                            {"alias", prms.alias}};
  exporter->add("TCtbI_RxPdg",  labels, MetricType::Gauge,   [&](){ return _transport.pending(); });
  exporter->add("TCtbI_BatCt",  labels, MetricType::Counter, [&](){ return _batchCount;          });
  exporter->add("TCtbI_EvtCt",  labels, MetricType::Counter, [&](){ return _eventCount;          });
  exporter->add("TCtbI_MisCt",  labels, MetricType::Counter, [&](){ return _missing;             });
  exporter->add("TCtbI_DefSz",  labels, MetricType::Counter, [&](){ return _deferred.size();     });
  exporter->add("TCtbI_BypCt",  labels, MetricType::Counter, [&](){ return _bypassCount;         });
  exporter->add("TCtbI_NPrgCt", labels, MetricType::Counter, [&](){ return _noProgCount;         });
}

EbCtrbInBase::~EbCtrbInBase()
{
  if (_region)  free(_region);
  _region = nullptr;
}

int EbCtrbInBase::resetCounters()
{
  _batchCount  = 0;
  _eventCount  = 0;
  _missing     = 0;
  _bypassCount = 0;

  _noProgCount = 0;
  _prvNPCnt    = 0;

  return 0;
}

void EbCtrbInBase::shutdown()
{
  _transport.shutdown();
}

void EbCtrbInBase::disconnect()
{
  for (auto link : _links)  _transport.disconnect(link);
  _links.clear();
}

void EbCtrbInBase::unconfigure()
{
}

int EbCtrbInBase::startConnection(std::string& port)
{
  int rc = _transport.listen(_prms.ifAddr, port, MAX_TEBS);
  if (rc)
  {
    logging::error("%s:\n  Failed to initialize %s EbLfServer on %s:%s",
                   __PRETTY_FUNCTION__, "TEB", _prms.ifAddr.c_str(), port.c_str());
    return rc;
  }

  return 0;
}

int EbCtrbInBase::connect(size_t resSizeGuess, unsigned numBuffers)
{
  unsigned numEbs = std::bitset<64>(_prms.builders).count();

  _links.resize(numEbs);

  int rc = linksConnect(_transport, _links, "TEB");
  if (rc)  return rc;

  // Assume an existing region is already appropriately sized, else make a guess
  // at a suitable RDMA region to avoid spending time in Configure.
  // If it's too small, it will be corrected during Configure
  if (!_region)                         // No need to guess again
  {
    // Make a guess at the size of the Result region
    size_t regSizeGuess = resSizeGuess * numBuffers;

    _region = allocRegion(regSizeGuess);
    if (!_region)
    {
      logging::error("%s:\n  "
                     "No memory found for Result MR for %s of size %zd",
                     __PRETTY_FUNCTION__, "TEB", regSizeGuess);
      return ENOMEM;
    }

    // Save the allocated size, which may be more than the required size
    _regSize       = regSizeGuess;
    _maxResultSize = resSizeGuess;
  }

  rc = _transport.setupMr(_region, _regSize);
  if (rc)  return rc;

  return 0;
}

int EbCtrbInBase::configure(unsigned numTebBuffers)
{
  // To give maximal chance of inspection with a debugger of a previous run's
  // information, clear it in configure() rather than in unconfigure()
  _inputs = nullptr;
  _deferred.clear();

  int rc = _linksConfigure(_links, _prms.id, numTebBuffers, "TEB");
  if (rc)  return rc;

  return 0;
}

int EbCtrbInBase::_linksConfigure(std::vector<EbLfSvrLink*>& links,
                                  unsigned                   id,
                                  unsigned                   numTebBuffers,
                                  const char*                peer)
{
  std::vector<EbLfSvrLink*> tmpLinks(links.size());
  size_t size = 0;

  // Since each EB handles a specific batch, one region can be shared by all
  for (auto link : links)
  {
    auto   t0(std::chrono::steady_clock::now());
    int    rc;
    size_t regSize;
    if ( (rc = link->prepare(id, &regSize, peer)) )
    {
      logging::error("%s:\n  Failed to prepare link with %s ID %d",
                     __PRETTY_FUNCTION__, peer, link->id());
      return rc;
    }
    unsigned rmtId  = link->id();
    tmpLinks[rmtId] = link;

    if (!size)
    {
      // Reallocate the region if the required size has changed.
      // The Results region size must match that on the TEB since it may produce
      // results batches that contain entries not meant for this particular
      // contributor (e.g., due to its being in a slower RoG) and these will
      // take up space not taken into account by the MemPool::nbuffers() value.
      if (regSize != _regSize)
      {
        if (_region)  free(_region);

        _region = allocRegion(regSize);
        if (!_region)
        {
          logging::error("%s:\n  "
                         "No memory found for Result MR for %s ID %d of size %zd",
                         __PRETTY_FUNCTION__, peer, rmtId, regSize);
          return ENOMEM;
        }

        _regSize = regSize;
      }
      _maxResultSize = regSize / numTebBuffers;
      size           = regSize;
    }
    else if (regSize != size)
    {
      logging::error("%s:\n  Results MR size (%zd) cannot vary between %ss "
                     "(%zd from Id %u)", __PRETTY_FUNCTION__, size, peer, regSize, rmtId);
      return -1;
    }

    if ( (rc = link->setupMr(_region, regSize, peer)) )
    {
      logging::error("%s:\n  Failed to set up Result MR for %s ID %d, "
                     "%p:%p, size %zd", __PRETTY_FUNCTION__, peer, rmtId,
                     _region, static_cast<char*>(_region) + regSize, regSize);
      return rc;
    }

    auto t1 = std::chrono::steady_clock::now();
    auto dT = std::chrono::duration_cast<ms_t>(t1 - t0).count();
    logging::info("Inbound   link with   %3s ID %2d configured in %4lu ms",
                  peer, rmtId, dT);
  }

  links = tmpLinks;                     // Now in remote ID sorted order

  return 0;
}

void EbCtrbInBase::receiver(TebContributor& ctrb, std::atomic<bool>& running)
{
  int rc = pinThread(pthread_self(), _prms.core[1]);
  if (rc && _prms.verbose)
  {
    logging::error("%s:\n  Error pinning thread to core %d:\n  %m",
                   __PRETTY_FUNCTION__, _prms.core[1]);
  }

  logging::info("Receiver thread is starting");

  int rcPrv = 0;
  while (running.load(std::memory_order_relaxed))
  {
    rc = _process(ctrb);
    if (rc < 0)
    {
      if (rc == -FI_ENOTCONN)
      {
        logging::critical("Receiver thread lost connection with a TEB");
        throw "Receiver thread lost connection with a TEB";
      }
      if (rc == rcPrv)  throw "Repeating fatal error";
    }
    rcPrv = rc;
  }

  logging::info("Receiver thread finished");
}

int EbCtrbInBase::_process(TebContributor& ctrb)
{
  int rc;

  // Pend for a Results batch (a set of EbDgrams) and process it.
  uint64_t  data;
  const int tmo = 100;                  // milliseconds
  if ( (rc = _transport.pend(&data, tmo)) < 0)
  {
    if (rc == -FI_ETIMEDOUT)
    {
      _matchUp(ctrb, nullptr);         // Try to sweep out any deferred Results
      rc = 0;
    }
    else if (_transport.pollEQ() == -FI_ENOTCONN)
      rc = -FI_ENOTCONN;
    else
      logging::error("%s:\n  pend() error %d (%s)\n",
                     __PRETTY_FUNCTION__, rc, strerror(-rc));
    return rc;
  }

  unsigned src = ImmData::src(data);
  unsigned idx = ImmData::idx(data);
  auto     lnk = _links[src];
  auto     ofs = idx * _maxResultSize;
  auto     bdg = static_cast<const ResultDgram*>(lnk->lclAdx(ofs)); // (char*)_region + ofs;

  if (UNLIKELY(_prms.verbose >= VL_BATCH))
  {
    auto     pid     = bdg->pulseId();
    unsigned ctl     = bdg->control();
    unsigned env     = bdg->env;
    auto&    pending = ctrb.pending();
    printf("CtrbIn  rcvd        %6lu result  [%8u] @ "
           "%16p, ctl %02x, pid %014lx, env %08x,            src %2u, empty %c, cnt %u, data %08lx\n",
           _batchCount, idx, bdg, ctl, pid, env, src, pending.is_empty() ? 'Y' : 'N',
           pending.guess_size(), data);
  }

  _matchUp(ctrb, bdg);

  ++_batchCount;

  return 0;
}

void EbCtrbInBase::_matchUp(TebContributor&    ctrb,
                            const ResultDgram* results)
{
  auto& pending = ctrb.pending();
  auto  inputs  = _inputs;                // Pick up where we left off

  if (results)  _defer(results);          // Defer Results batch

  while (true)
  {
    if (!inputs && !pending.peek(inputs)) // If there are no left-over Inputs, and
      break;                              // no new Inputs batch, nothing to do

    if (UNLIKELY(!(inputs->readoutGroups() & (1 << _prms.partition))))
    {                                     // Common RoG didn't trigger
      _deliverBypass(ctrb, inputs);       // Handle bypass event
      continue;
    }

    if (_deferred.empty())  break;        // If no Results batch, nothing to do
    results = _deferred.front();          // Get Results batch

    const auto res = results;
    const auto inp = inputs;
    _deliver(ctrb, results, inputs);      // Handle Results batch
    if (!inputs)                          // If Inputs batch was consummed
    {
      const EbDgram* ins;
      pending.try_pop(ins);               // Take Inputs batch off the list
    }
    _deferred.pop_front();                // Dequeue the deferred Results batch
    if (results)                          // If not all deferred Results were consummed
      _defer(results);                    //   defer the remainder

    // No progress can legitimately happen with multiple TEBs presenting events
    // out of order.  These will be deferred so that when the expected Result
    // arrives (according to the Input), it will be handled in the proper order
    if ((results == res) && (inputs == inp))
    {
      //if (_noProgCount - _prvNPCnt < 5)
      //{
      //  unsigned rIdx = (reinterpret_cast<const char*>(res) -
      //                   static_cast<const char*>(_region)) / _maxResultSize;
      //  unsigned iIdx = ctrb.index(inp);
      //  printf("*** No progress %ld: res %u %014lx %s, inp %u %014lx %s\n",
      //         _noProgCount - _prvNPCnt,
      //         rIdx, res->pulseId(), TransitionId::name(res->service()),
      //         iIdx, inp->pulseId(), TransitionId::name(inp->service()));
      //  _dump(ctrb, results, inputs);
      //  printf("deferred:\n");
      //  for (auto batch : _deferred)
      //  {
      //    unsigned index = (reinterpret_cast<const char*>(batch) -
      //                      static_cast<const char*>(_region)) / _maxResultSize;
      //    printf("  %014lx %s:\n", batch->pulseId(), TransitionId::name(batch->service()));
      //    dumpBatch(ctrb, batch, _maxResultSize, index);
      //  }
      //}
      ++_noProgCount;
      break;                              // Exit loop on no progress
    }
    _prvNPCnt = _noProgCount;
  }                                       // Loop for a newer Inputs batch
  _inputs = inputs;                       // Save any remaining Inputs for next time
}

void EbCtrbInBase::_defer(const ResultDgram* results)
{

  for (auto it = _deferred.begin(); it != _deferred.end(); ++it)
  {
    auto batch = *it;
    assert(results->pulseId() != batch->pulseId());
    if (results->pulseId() < batch->pulseId())
    {
      _deferred.insert(it, results);    // This inserts before
      return;
    }
  }
  _deferred.push_back(results);
}

void EbCtrbInBase::_deliverBypass(TebContributor& ctrb,
                                  const EbDgram*& inputs)
{
  auto pid = inputs->pulseId();
  ResultDgram result(*inputs, 0);       // Bypass events are not monitorable (no buffer # received via TEB)
  result.persist(true);                 // Always record bypass events
  result.setEOL();
  const ResultDgram* results = &result;

  _deliver(ctrb, results, inputs);
  assert(!results && !inputs);

  ++_bypassCount;
  ++_eventCount;

  const EbDgram* ins;
  ctrb.pending().try_pop(ins);          // Take Inputs batch off the list
  assert (ins->pulseId() == pid);
}

void EbCtrbInBase::_deliver(TebContributor&     ctrb,
                            const ResultDgram*& results,
                            const EbDgram*&     inputs)
{
  auto       result  = results;
  auto       input   = inputs;
  const auto rSize   = _maxResultSize;
  const auto iSize   = _prms.maxInputSize;
  auto       rPid    = result->pulseId();
  auto       iPid    = input->pulseId();
  unsigned   idx     = ctrb.index(inputs);
  unsigned   missing = _missing;

  // This code expects to handle events in pulse ID order
  while (rPid <= iPid)
  {
    uint64_t rPidPrv = 0;
    if (UNLIKELY(!(rPid > rPidPrv)))
    {
      logging::critical("%s:\n  rPid %014lx <= rPidPrv %014lx\n",
                        __PRETTY_FUNCTION__, rPid, rPidPrv);
      _dump(ctrb, results, inputs);
      throw "Result pulse ID didn't advance";
    }
    rPidPrv = rPid;

    // Ignore Results for which there is no Input
    // This can happen due to this DRP being in a different readout group than
    // the one for which the result is for, or somehow having missed a
    // contribution that was subsequently fixed up by the TEB.  In both cases
    // there is validly no Input corresponding to the Result.

    if (UNLIKELY(_prms.verbose >= VL_EVENT))
    {
      auto rTs    = result->time.value();
      auto env    = result->env;
      auto src    = result->xtc.src.value();
      auto ctl    = result->control();
      auto svc    = TransitionId::name(result->service());
      auto extent = sizeof(*result) + result->xtc.sizeofPayload();
      printf("CtrbIn  found  %15s  [%8u]    @ "
             "%16p, ctl %02x, ts %016lx, pid %014lx, env %08x, sz %6zd, TEB %2u, dlvr %c [%014lx], res %08x, %08x \n",
             svc, idx, result, ctl, rTs, rPid, env, extent, src, rPid == iPid ? 'Y' : 'N', iPid, result->data(), result->monBufNo());
    }

    if (rPid == iPid)
    {
      static uint64_t iPidPrv = 0;
      if (UNLIKELY(!(iPid > iPidPrv)))
      {
        logging::critical("%s:\n  iPid %014lx <= iPidPrv %014lx\n",
                          __PRETTY_FUNCTION__, iPid, iPidPrv);
        _dump(ctrb, results, inputs);
        throw "Input pulse ID didn't advance";
      }
      iPidPrv = iPid;

      process(*result, idx++);

      ++_eventCount;

      if (input->isEOL())
      {
        inputs  = nullptr;
        results = result->isEOL() ? nullptr : reinterpret_cast<const ResultDgram*>(reinterpret_cast<const char*>(result) + rSize);
        return;
      }

      input = reinterpret_cast<const EbDgram*>(reinterpret_cast<const char*>(input) + iSize);

      iPid = input->pulseId();
    }
    else
    {
      assert (rPid < iPid);        // rPid > iPid would have caused loop to end

      if (result->readoutGroups() & _prms.readoutGroup) // Is a match expected?
      {
        if (!(result->readoutGroups() & _prms.contractor)) // Are we Receiver-only?
        {
          // In this case, the Result could have appeared before the Input was
          // prepared, so the Result should be deferred and timed out
          // For now, continue on
          // ToDo: Implement timing/sweeping out missing Input

          ++_missing;

          logging::error("%s:\n  No Input found for Result %014lx",
                         __PRETTY_FUNCTION__, rPid);
          _dump(ctrb, results, inputs);
        }
        // else Result was fixed up and Input is missing, so discard Result
      }
      // else no match is expected, so discard Result
    }

    if (result->isEOL())
    {
      inputs  = input;
      results = nullptr;
      return;
    }

    result = reinterpret_cast<const ResultDgram*>(reinterpret_cast<const char*>(result) + rSize);

    rPid = result->pulseId();
  }

  // No Result for Input is allowed to happen when it is too new for the Input,
  // as can happen when multiple TEBs present Results out of order
  //logging::error("%s:\n  No Result found for Input %014lx",
  //               __PRETTY_FUNCTION__, iPid);
  if (_missing != missing)
  {
    logging::error("%s:\n  No Inputs found for %d Results",
                   __PRETTY_FUNCTION__, _missing - missing);
    _dump(ctrb, results, inputs);
  }

  inputs  = input;
  results = result;
}

void EbCtrbInBase::_dump(TebContributor&    ctrb,
                         const ResultDgram* results,
                         const EbDgram*     inputs) const
{
  if (results)
  {
    unsigned index = (reinterpret_cast<const char*>(results) -
                      static_cast<const char*>(_region)) / _maxResultSize;
    printf("Results:\n");
    dumpBatch(ctrb, results, _maxResultSize, index);
  }

  if (inputs)
  {
    printf("Inputs:\n");
    dumpBatch(ctrb, inputs, _prms.maxInputSize, ctrb.index(inputs));
  }
}
