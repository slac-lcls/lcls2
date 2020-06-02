#include "EbCtrbInBase.hh"

#include "Endpoint.hh"
#include "EbLfServer.hh"
#include "Batch.hh"
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


using namespace XtcData;
using namespace Pds;
using namespace Pds::Eb;
using logging  = psalg::SysLog;


static void dumpBatch(const TebContributor& ctrb,
                      const EbDgram*        batch,
                      const size_t          size)
{
  auto dg   = batch;
  auto bPid = dg->pulseId();
  for (unsigned i = 0; i < MAX_ENTRIES; ++i)
  {
    auto pid = dg->pulseId();
    auto svc = TransitionId::name(dg->service());
    auto rog = dg->readoutGroups();
    auto dmg = dg->xtc.damage.value();
    printf("  %2d: %15s, pid %014lx, diff %016lx, RoG %2hx, dmg %04x, appPrm %p %s\n",
           i, svc, pid, pid - bPid, rog, dmg, ctrb.retrieve(pid), dg->isEOL() ? "EOL" : "");
    if (dg->isEOL())  return;
    dg = reinterpret_cast<const EbDgram*>(reinterpret_cast<const char*>(dg) + size);
  }
  printf("  EOL not found!\n");
}


EbCtrbInBase::EbCtrbInBase(const TebCtrbParams&                   prms,
                           const std::shared_ptr<MetricExporter>& exporter) :
  _transport    (prms.verbose),
  _links        (),
  _maxResultSize(0),
  _batchCount   (0),
  _eventCount   (0),
  _missing      (0),
  _bypassCount  (0),
  _prms         (prms),
  _region       (nullptr)
{
  std::map<std::string, std::string> labels{{"instrument", prms.instrument},{"partition", std::to_string(prms.partition)}};
  exporter->add("TCtbI_RxPdg", labels, MetricType::Gauge,   [&](){ return _transport.pending(); });
  exporter->add("TCtbI_BatCt", labels, MetricType::Counter, [&](){ return _batchCount;          });
  exporter->add("TCtbI_EvtCt", labels, MetricType::Counter, [&](){ return _eventCount;          });
  exporter->add("TCtbI_MisCt", labels, MetricType::Counter, [&](){ return _missing;             });
  exporter->add("TCtbI_DefSz", labels, MetricType::Counter, [&](){ return _deferred.size();     });
  exporter->add("TCtbI_BypCt", labels, MetricType::Counter, [&](){ return _bypassCount;         });
}

int EbCtrbInBase::configure(const TebCtrbParams& prms)
{
  _batchCount   = 0;
  _eventCount   = 0;
  _missing      = 0;
  _bypassCount  = 0;

  unsigned numEbs = std::bitset<64>(prms.builders).count();
  _links.resize(numEbs);

  int rc;
  if ( (rc = _transport.initialize(prms.ifAddr, prms.port, numEbs)) )
  {
    logging::error("%s:\n  Failed to initialize EbLfServer on %s:%s",
                   __PRETTY_FUNCTION__, prms.ifAddr, prms.port);
    return rc;
  }

  size_t size = 0;

  // Since each EB handles a specific batch, one region can be shared by all
  for (unsigned i = 0; i < _links.size(); ++i)
  {
    EbLfSvrLink*   link;
    const unsigned tmo(120000);         // Milliseconds
    if ( (rc = _transport.connect(&link, prms.id, tmo)) )
    {
      logging::error("%s:\n  Error connecting to TEB %d",
                     __PRETTY_FUNCTION__, i);
      return rc;
    }
    unsigned rmtId = link->id();
    _links[rmtId] = link;

    logging::debug("Inbound link with TEB ID %d connected", rmtId);

    size_t regSize;
    if ( (rc = link->prepare(&regSize)) )
    {
      logging::error("%s:\n  Failed to prepare link with TEB ID %d",
                     __PRETTY_FUNCTION__, rmtId);
      return rc;
    }

    if (!size)
    {
      size           = regSize;
      _maxResultSize = regSize / (MAX_BATCHES * MAX_ENTRIES);

      _region = allocRegion(regSize);
      if (_region == nullptr)
      {
        logging::error("%s:\n  No memory found for a Result MR of size %zd",
                       __PRETTY_FUNCTION__, regSize);
        return ENOMEM;
      }
    }
    else if (regSize != size)
    {
      logging::error("%s:\n  Error: Result MR size (%zd) cannot vary between TEBs "
                     "(%zd from Id %d)", __PRETTY_FUNCTION__, size, regSize, rmtId);
      return -1;
    }

    if ( (rc = link->setupMr(_region, regSize)) )
    {
      char* region = static_cast<char*>(_region);
      logging::error("%s:\n  Failed to set up Result MR for TEB ID %d, %p:%p, size %zd",
                     __PRETTY_FUNCTION__, rmtId, region, region + regSize, regSize);
      if (_region)  free(_region);
      _region = nullptr;
      return rc;
    }
    if (link->postCompRecv())
    {
      logging::warning("%s:\n  Failed to post CQ buffers for DRP ID %d",
                       __PRETTY_FUNCTION__, rmtId);
    }

    logging::info("Inbound link with TEB ID %d connected and configured", rmtId);
  }

  return 0;
}

void EbCtrbInBase::receiver(TebContributor& ctrb, std::atomic<bool>& running)
{
  int rc = pinThread(pthread_self(), _prms.core[1]);
  if (rc && _prms.verbose)
  {
    logging::error("%s:\n  Error from pinThread:\n  %s",
                   __PRETTY_FUNCTION__, strerror(rc));
  }

  logging::info("Receiver thread is starting");

  _inputs = nullptr;

  while (running.load(std::memory_order_relaxed))
  {
    if (_process(ctrb) < 0)
    {
      if (_transport.pollEQ() == -FI_ENOTCONN)
      {
        logging::critical("Receiver thread lost connection");
        break;
      }
    }
  }

  _shutdown();

  logging::info("Receiver thread finished");
}

void EbCtrbInBase::_shutdown()
{
  for (auto it = _links.begin(); it != _links.end(); ++it)
  {
    _transport.disconnect(*it);
  }
  _links.clear();
  _transport.shutdown();

  if (_region)  free(_region);
  _region = nullptr;
}

int EbCtrbInBase::_process(TebContributor& ctrb)
{
  int rc;

  // Pend for a Results batch (a set of EbDgrams) and process it.
  uint64_t  data;
  const int tmo = 100;                  // milliseconds
  if ( (rc = _transport.pend(&data, tmo)) < 0)
  {
    // Try to sweep out any deferred Results
    if (rc == -FI_ETIMEDOUT)  _matchUp(ctrb, nullptr);
    return rc;
  }

  unsigned src = ImmData::src(data);
  unsigned idx = ImmData::idx(data);
  auto     lnk = _links[src];
  auto     bdg = static_cast<const ResultDgram*>(lnk->lclAdx(idx * _maxResultSize));
  auto     pid = bdg->pulseId();
  if (lnk->postCompRecv())
  {
    logging::warning("%s:\n  Failed to post CQ buffers for DRP ID %d",
                     __PRETTY_FUNCTION__, src);
  }

  if (unlikely(_prms.verbose >= VL_BATCH))
  {
    unsigned ctl     = bdg->control();
    unsigned env     = bdg->env;
    auto&    pending = ctrb.pending();
    printf("CtrbIn  rcvd        %6ld result  [%8d] @ "
           "%16p, ctl %02x, pid %014lx, env %08x,            src %2d, empty %c, cnt %d\n",
           _batchCount, idx, bdg, ctl, pid, env, lnk->id(), pending.is_empty() ? 'Y' : 'N',
           pending.guess_size());
  }

  _matchUp(ctrb, bdg);

  ++_batchCount;

  return 0;
}

void EbCtrbInBase::_matchUp(TebContributor&    ctrb,
                            const ResultDgram* results)
{
  auto& pending = ctrb.pending();
  auto  inputs  = _inputs;

  if (results)  _defer(results);          // Defer Results batch

  while (true)
  {
    if (!inputs && !pending.peek(inputs)) // If there are no left-over Inputs, and
      break;                              // no new Inputs batch, nothing to do

    if (unlikely(!(inputs->readoutGroups() & (1 << _prms.partition))))
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
      ctrb.release(ins->pulseId());       // Release the Inputs batch
    }
    _deferred.pop_front();                // Dequeue the deferred Results batch
    if (results)                          // If not all deferred Results were consummed
      _defer(results);                    //   defer the remainder

    // Revisit: This probably doesn't happen anymore, but be alert for it for while
    if ((results == res) && (inputs == inp))
    {
      printf("No progress: res %014lx, inp %014lx\n", res->pulseId(), inp->pulseId());
      break;                              // Revisit: Break if no progress
    }
  }                                       // Loop for a newer Inputs batch
  _inputs = inputs;                       // Save any remaining Inputs for next time
}

void EbCtrbInBase::_defer(const ResultDgram* results)
{

  for (auto it = _deferred.begin(); it != _deferred.end(); ++it)
  {
    auto batch = *it;
    assert (results->pulseId() != batch->pulseId());
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
  unsigned line = 0;                    // Revisit: For future expansion
  result.persist(line, true);           // Always record bypass events
  result.setEOL();
  const ResultDgram* results = &result;

  _deliver(ctrb, results, inputs);
  assert (!results && !inputs);

  ++_bypassCount;

  const EbDgram* ins;
  ctrb.pending().try_pop(ins);          // Take Inputs batch off the list
  assert (ins->pulseId() == pid);
  ctrb.release(pid);                    // Release the Inputs batch
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
  unsigned   missing = 0;

  // This code expects to handle events in pulse ID order
  while (rPid <= iPid)
  {
    // Revisit: Why does this fail?
    //static uint64_t rPidPrv = 0;  assert (rPid > rPidPrv);  rPidPrv = rPid;

    // Ignore Results for which there is no Input
    // This can happen due to this DRP being in a different readout group than
    // the one for which the result is for, or somehow having missed a
    // contribution that was subsequently fixed up by the TEB.  In both cases
    // there is validly no Input corresponding to the Result.

    if (unlikely(_prms.verbose >= VL_EVENT))
    {
      auto idx    = Batch::index(iPid);
      auto env    = result->env;
      auto src    = result->xtc.src.value();
      auto ctl    = result->control();
      auto svc    = TransitionId::name(result->service());
      auto extent = sizeof(*result) + result->xtc.sizeofPayload();
      printf("CtrbIn  found  %15s  [%8ld]    @ "
             "%16p, ctl %02x, pid %014lx, env %08x, sz %6zd, TEB %2d, deliver %c [%014lx]\n",
             svc, idx, result, ctl, rPid, env, extent, src, rPid == iPid ? 'Y' : 'N', iPid);
    }

    if (rPid == iPid)
    {
      static uint64_t iPidPrv = 0;  assert (iPid > iPidPrv);  iPidPrv = iPid;

      process(*result, ctrb.retrieve(iPid));

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
    else if (result->readoutGroups() & _prms.readoutGroup) // Is a match expected?
    {
      // A fixed-up event for which this DRP didn't supply input is allowed, else count
      if (!(result->xtc.damage.value() & (1 <<  Damage::DroppedContribution)))
        ++_missing;
    }

    if (result->isEOL())
    {
      if ((input == inputs) && _missing)
      {
        if (rPid < iPid)
          logging::error("%s:\n  Results %014lx too old for Inputs %014lx",
                         __PRETTY_FUNCTION__, results->pulseId(), inputs->pulseId());
        else
          logging::error("%s:\n  No Inputs found for %u Results",
                         __PRETTY_FUNCTION__, _missing);
        _dump(ctrb, results, inputs);
      }
      inputs  = input;
      results = nullptr;
      return;
    }

    result = reinterpret_cast<const ResultDgram*>(reinterpret_cast<const char*>(result) + rSize);

    rPid = result->pulseId();
  }

  logging::error("%s:\n  No Result found for Input %014lx",
                 __PRETTY_FUNCTION__, iPid);
  if (_missing)
    logging::error("%s:\n  No Inputs found for %u Results",
                   __PRETTY_FUNCTION__, missing);
  _dump(ctrb, results, inputs);

  inputs  = input;
  results = result;
}

void EbCtrbInBase::_dump(TebContributor&    ctrb,
                         const ResultDgram* results,
                         const EbDgram*     inputs) const
{
  if (results)
  {
    printf("Results:\n");
    dumpBatch(ctrb, results, _maxResultSize);
  }

  if (inputs)
  {
    printf("Inputs:\n");
    dumpBatch(ctrb, inputs, _prms.maxInputSize);
  }
}
