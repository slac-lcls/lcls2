#include "EbCtrbInBase.hh"

#include "Endpoint.hh"
#include "EbLfServer.hh"
#include "Batch.hh"
#include "TebContributor.hh"
#include "ResultDgram.hh"

#include "utilities.hh"

#include "psalg/utils/SysLog.hh"
#include "psdaq/service/MetricExporter.hh"
#include "xtcdata/xtc/Dgram.hh"

#ifdef NDEBUG
//#undef NDEBUG
#endif

#include <cassert>
#include <string>
#include <bitset>
#include <chrono>


using namespace XtcData;
using namespace Pds::Eb;
using logging  = psalg::SysLog;


EbCtrbInBase::EbCtrbInBase(const TebCtrbParams&                   prms,
                           const std::shared_ptr<MetricExporter>& exporter) :
  _transport   (prms.verbose),
  _links       (),
  _maxBatchSize(0),
  _batchCount  (0),
  _eventCount  (0),
  _deliverCount(0),
  _prms        (prms),
  _region      (nullptr)
{
  std::map<std::string, std::string> labels{{"partition", std::to_string(prms.partition)}};
  exporter->add("TCtbI_RxPdg", labels, MetricType::Gauge,   [&](){ return _transport.pending(); });
  exporter->add("TCtbI_BatCt", labels, MetricType::Counter, [&](){ return _batchCount;          });
  exporter->add("TCtbI_EvtCt", labels, MetricType::Counter, [&](){ return _eventCount;          });
  exporter->add("TCtbI_DlrCt", labels, MetricType::Counter, [&](){ return _deliverCount;        });
}

int EbCtrbInBase::configure(const TebCtrbParams& prms)
{
  _batchCount   = 0;
  _eventCount   = 0;
  _deliverCount = 0;

  unsigned numEbs = std::bitset<64>(prms.builders).count();
  _links.resize(numEbs);

  int rc;
  if ( (rc = _transport.initialize(prms.ifAddr, prms.port, numEbs)) )
  {
    logging::error("%s:\n  Failed to initialize EbLfServer on %s:%s\n",
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
      logging::error("%s:\n  Error connecting to TEB %d\n",
                     __PRETTY_FUNCTION__, i);
      return rc;
    }
    unsigned rmtId = link->id();
    _links[rmtId] = link;

    logging::debug("Inbound link with TEB ID %d connected\n", rmtId);

    size_t regSize;
    if ( (rc = link->prepare(&regSize)) )
    {
      logging::error("%s:\n  Failed to prepare link with TEB ID %d\n",
                     __PRETTY_FUNCTION__, rmtId);
      return rc;
    }

    if (!size)
    {
      size          = regSize;
      _maxBatchSize = regSize / MAX_BATCHES;

      _region = allocRegion(regSize);
      if (_region == nullptr)
      {
        logging::error("%s:\n  No memory found for a Result MR of size %zd\n",
                       __PRETTY_FUNCTION__, regSize);
        return ENOMEM;
      }
    }
    else if (regSize != size)
    {
      logging::error("%s:\n  Error: Result MR size (%zd) cannot differ between TEBs "
                     "(%zd from Id %d)\n", __PRETTY_FUNCTION__, size, regSize, rmtId);
      return -1;
    }

    if ( (rc = link->setupMr(_region, regSize)) )
    {
      char* region = static_cast<char*>(_region);
      logging::error("%s:\n  Failed to set up Result MR for TEB ID %d, %p:%p, size %zd\n",
                     __PRETTY_FUNCTION__, rmtId, region, region + regSize, regSize);
      if (_region)  free(_region);
      _region = nullptr;
      return rc;
    }
    if (link->postCompRecv())
    {
      logging::warning("%s:\n  Failed to post CQ buffers for DRP ID %d\n",
                       __PRETTY_FUNCTION__, rmtId);
    }

    logging::info("Inbound link with TEB ID %d connected and configured\n", rmtId);
  }

  return 0;
}

void EbCtrbInBase::receiver(TebContributor& ctrb, std::atomic<bool>& running)
{
  int rc = pinThread(pthread_self(), _prms.core[1]);
  if (rc && _prms.verbose)
  {
    logging::error("%s:\n  Error from pinThread:\n  %s\n",
                   __PRETTY_FUNCTION__, strerror(rc));
  }

  logging::info("Receiver thread is starting\n");

  while (running.load(std::memory_order_relaxed))
  {
    if (_process(ctrb) < 0)
    {
      if (_transport.pollEQ() == -FI_ENOTCONN)  break;
    }
  }

  _shutdown();

  logging::info("Receiver thread is exiting\n");
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

  // Pend for a result batch (a set of EbDgrams) and process it.
  uint64_t  data;
  const int tmo = 100;                  // milliseconds
  if ( (rc = _transport.pend(&data, tmo)) < 0)  return rc;

  unsigned           src = ImmData::src(data);
  unsigned           idx = ImmData::idx(data);
  EbLfSvrLink*       lnk = _links[src];
  const ResultDgram* bdg = reinterpret_cast<const ResultDgram*>(lnk->lclAdx(idx * _maxBatchSize));
  uint64_t           pid = bdg->pulseId();
  if ( (rc = lnk->postCompRecv()) )
  {
    logging::warning("%s:\n  Failed to post CQ buffers for DRP ID %d\n",
                     __PRETTY_FUNCTION__, src);
  }

  if (_prms.verbose >= VL_BATCH)
  {
    unsigned   ctl     = bdg->control();
    unsigned   env     = bdg->env;
    BatchFifo& pending = ctrb.pending();
    printf("CtrbIn  rcvd        %6ld result  [%8d] @ "
           "%16p, ctl %02x, pid %014lx, env %08x,            src %2d, empty %c, cnt %zd, result %p\n",
           _batchCount, idx, bdg, ctl, pid, env, lnk->id(), pending.empty() ? 'Y' : 'N',
           pending.count(), ctrb.batch(idx)->result());
  }

  _pairUp(ctrb, idx, bdg);

  ++_batchCount;

  return 0;
}

#define unlikely(expr)  __builtin_expect(!!(expr), 0)
#define likely(expr)    __builtin_expect(!!(expr), 1)

void EbCtrbInBase::_pairUp(TebContributor&    ctrb,
                           unsigned           idx,
                           const ResultDgram* result)
{
  BatchFifo& pending = ctrb.pending();

  if (pending.empty())
  {
    auto t0 = fast_monotonic_clock::now();
    do
    {
      std::this_thread::yield();

      using     ms_t  = std::chrono::milliseconds;
      const int msTmo = 100;
      auto      t1    = fast_monotonic_clock::now();

      if (std::chrono::duration_cast<ms_t>(t1 - t0).count() > msTmo)
      {
        logging::warning("%s:\n  No Input batch for Result: empty pending FIFO timeout:\n"
                         "    new result   idx %08x, pid %014lx\n", __PRETTY_FUNCTION__,
                         idx, result->pulseId());
        return;
      }
    }
    while (pending.empty());
  }

  const Batch* inputs = pending.front();
  if (inputs->index() != idx)
  {
    Batch* batch = ctrb.batch(idx);

    if (unlikely(batch->result()))
    {
      logging::critical("%s:\n  Unhandled result found:\n"
                        "    new result      idx %08x, pid %014lx\n"
                        "    looked up batch idx %08x, pid %014lx\n"
                        "    unhandled result              pid %014lx\n"
                        "    pending head    idx %08x, pid %014lx\n",
                        __PRETTY_FUNCTION__,
                        idx, result->pulseId(),
                        batch->index(), batch->id(),
                        batch->result()->pulseId(),
                        inputs->index(), inputs->id());
      abort();
    }

    batch->result(result);            // Batch is possibly not allocated yet

    result = inputs->result();        // Go on to proccess whatever is ready
  }

  while (result)
  {
    uint64_t iPid = inputs->id();
    uint64_t rPid = result->pulseId();
    if (unlikely((iPid ^ rPid) & ~(BATCH_DURATION - 1))) // Include bits above index()
    {
      logging::critical("%s:\n  Result / Input batch mismatch: "
                        "Input pid %014lx, Result pid %014lx, xor %014lx, diff %ld\n",
                        __PRETTY_FUNCTION__, iPid, rPid, iPid ^ rPid, iPid - rPid);
      abort();
    }

    _deliver(ctrb, result, inputs);

    const Batch* ins;
    pending.pop(ins);
    //pending.pop();
    ctrb.release(inputs);      // Release input, and by proxy, result batches
    if (pending.empty())  break;

    inputs = pending.front();
    result = inputs->result();
  }
}

void EbCtrbInBase::_deliver(TebContributor&    ctrb,
                            const ResultDgram* results,
                            const Batch*       inputs)
{
  const ResultDgram* result = results;
  const EbDgram*     input  = static_cast<const EbDgram*>(inputs->buffer());
  const size_t       iSize  = inputs->size();
  const size_t       rSize  = _maxBatchSize / MAX_ENTRIES;
  uint64_t           rPid   = result->pulseId();
  uint64_t           iPid   = input->pulseId();
  unsigned           rCnt   = MAX_ENTRIES;
  unsigned           iCnt   = MAX_ENTRIES;
  while (true)
  {
    // Ignore results for which there is no input
    // This can happen due to this DRP being in a different readout group than
    // the one for which the result is for, or having missed a contribution that
    // was subsequently fixed up by the TEB.  In both cases there is no
    // input corresponding to the result.

    if (_prms.verbose >= VL_EVENT)
    {
      unsigned    idx    = inputs->index();
      unsigned    env    = result->env;
      unsigned    src    = result->xtc.src.value();
      unsigned    ctl    = result->control();
      const char* svc    = TransitionId::name(result->service());
      size_t      extent = sizeof(*result) + result->xtc.sizeofPayload();
      printf("CtrbIn  found  %15s  [%8d]    @ "
             "%16p, ctl %02x, pid %014lx, env %08x, sz %6zd, TEB %2d, deliver %c [%014lx]\n",
             svc, idx, result, ctl, rPid, env, extent, src, rPid == iPid ? 'Y' : 'N', iPid);
    }

    ++_eventCount;

    if (rPid == iPid)
    {
      process(*result, inputs->retrieve(iPid));

      ++_deliverCount;                  // Don't count events not meant for us

      if (!--iCnt || input->isEOL())  break; // Handle full list faster

      input = reinterpret_cast<const EbDgram*>(reinterpret_cast<const char*>(input) + iSize);

      iPid = input->pulseId();
    }

    if (!--rCnt || result->isEOL())  break; // Handle full list faster

    result = reinterpret_cast<const ResultDgram*>(reinterpret_cast<const char*>(result) + rSize);

    rPid = result->pulseId();
  }

  if (iCnt && !input->isEOL())
  {
    logging::warning("%s:\n  Not all Inputs received Results, inp: %d %014lx res: %d %014lx\n",
                     __PRETTY_FUNCTION__, MAX_ENTRIES - iCnt, iPid, MAX_ENTRIES - rCnt, rPid);
    printf("Results:\n");
    result = results;
    for (unsigned i = 0; i < MAX_ENTRIES; ++i)
    {
      uint64_t pid = result->pulseId();
      printf("  %2d: pid %014lx, appPrm %p\n", i, pid, inputs->retrieve(pid));
      if (result->isEOL())  break;
      result = reinterpret_cast<const ResultDgram*>(reinterpret_cast<const char*>(result) + rSize);
    }
    printf("Inputs:\n");
    inputs->dump();
  }
}
