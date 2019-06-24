#include "EbCtrbInBase.hh"

#include "Endpoint.hh"
#include "EbLfServer.hh"
#include "Batch.hh"
#include "TebContributor.hh"

#include "utilities.hh"

#include "psdaq/service/MetricExporter.hh"
#include "xtcdata/xtc/Dgram.hh"

#ifdef NDEBUG
//#undef NDEBUG
#endif

#include <cassert>
#include <string>
#include <bitset>


using namespace XtcData;
using namespace Pds::Eb;


EbCtrbInBase::EbCtrbInBase(const TebCtrbParams&            prms,
                           std::shared_ptr<MetricExporter> exporter) :
  _transport   (prms.verbose),
  _links       (),
  _maxBatchSize(0),
  _batchCount  (0),
  _eventCount  (0),
  _prms        (prms),
  _region      (nullptr)
{
  std::map<std::string, std::string> labels{{"partition", std::to_string(prms.partition)}};
  exporter->add("TCtbI_RxPdg", labels, MetricType::Gauge,   [&](){ return _transport.pending(); });
  exporter->add("TCtbI_BatCt", labels, MetricType::Counter, [&](){ return _batchCount;          });
  exporter->add("TCtbI_EvtCt", labels, MetricType::Counter, [&](){ return _eventCount;          });
}

int EbCtrbInBase::connect(const TebCtrbParams& prms)
{
  int rc;

  _batchCount = 0;
  _eventCount = 0;

  unsigned numEbs = std::bitset<64>(prms.builders).count();
  _links.resize(numEbs);

  if ( (rc = _transport.initialize(prms.ifAddr, prms.port, numEbs)) )
  {
    fprintf(stderr, "%s:\n  Failed to initialize EbLfServer\n",
            __PRETTY_FUNCTION__);
    return rc;
  }

  size_t size = 0;

  // Since each EB handles a specific batch, one region can be shared by all
  for (unsigned i = 0; i < numEbs; ++i)
  {
    EbLfLink*      link;
    const unsigned tmo(120000);         // Milliseconds
    if ( (rc = _transport.connect(&link, tmo)) )
    {
      fprintf(stderr, "%s:\n  Error connecting to TEB %d\n",
              __PRETTY_FUNCTION__, i);
      return rc;
    }

    size_t regSize;
    if ( (rc = link->preparePender(prms.id, &regSize)) )
    {
      fprintf(stderr, "%s:\n  Failed to prepare link with TEB %d\n",
              __PRETTY_FUNCTION__, i);
      return rc;
    }
    _links[link->id()] = link;

    if (!size)
    {
      size          = regSize;
      _maxBatchSize = regSize / MAX_BATCHES;

      _region = allocRegion(regSize);
      if (_region == nullptr)
      {
        fprintf(stderr, "%s:\n  No memory found for a Result MR of size %zd\n",
                __PRETTY_FUNCTION__, regSize);
        return ENOMEM;
      }
    }
    else if (regSize != size)
    {
      fprintf(stderr, "%s:\n  Error: Result MR size (%zd) cannot differ between TEBs "
              "(%zd from Id %d)\n", __PRETTY_FUNCTION__, size, regSize, link->id());
      return -1;
    }

    if ( (rc = link->setupMr(_region, regSize)) )
    {
      char* region = static_cast<char*>(_region);
      fprintf(stderr, "%s:\n  Failed to set up Result MR for TEB ID %d, %p:%p, size %zd\n",
              __PRETTY_FUNCTION__, link->id(), region, region + regSize, regSize);
      return rc;
    }
    link->postCompRecv();

    printf("Inbound link with TEB ID %d connected\n", link->id());
  }

  return 0;
}

void EbCtrbInBase::shutdown()
{
  for (auto it = _links.begin(); it != _links.end(); ++it)
  {
    _transport.shutdown(*it);
  }
  _links.clear();
  _transport.shutdown();

  if (_region)  free(_region);
  _region = nullptr;
}

void EbCtrbInBase::receiver(TebContributor& ctrb, std::atomic<bool>& running)
{
  pinThread(pthread_self(), _prms.core[1]);

  while (true)
  {
    if (!running)
    {
      if (_transport.pollEQ() == -FI_ENOTCONN)  break;
    }

    if (process(ctrb) < 0)
    {
      if (_transport.pollEQ() == -FI_ENOTCONN)  break;
    }
  }

  shutdown();
}

int EbCtrbInBase::process(TebContributor& ctrb)
{
  int rc;

  // Pend for a result batch (a set of Dgrams) and process it.
  uint64_t  data;
  const int tmo = 100;                  // milliseconds
  if ( (rc = _transport.pend(&data, tmo)) < 0)  return rc;

  ++_batchCount;

  unsigned     src = ImmData::src(data);
  unsigned     idx = ImmData::idx(data);
  EbLfLink*    lnk = _links[src];
  const Dgram* bdg = (const Dgram*)(lnk->lclAdx(idx * _maxBatchSize));
  uint64_t     pid = bdg->seq.pulseId().value();
  if ( (rc = lnk->postCompRecv()) )
  {
    fprintf(stderr, "%s:\n  Failed to post CQ buffers: %d\n",
            __PRETTY_FUNCTION__, rc);
  }

  if (_prms.verbose)
  {
    unsigned   ctl     = bdg->seq.pulseId().control();
    unsigned   env     = bdg->env;
    BatchFifo& pending = ctrb.pending();
    printf("CtrbIn  rcvd        %6ld result  [%5d] @ "
           "%16p, ctl %02x, pid %014lx,          src %2d, env %08x, E %d %zd, result %p\n",
           _batchCount, idx, bdg, ctl, pid, lnk->id(), env, pending.empty(),
           pending.count(), ctrb.batch(idx)->result());
  }

  _pairUp(ctrb, idx, bdg);

  return 0;
}

#define unlikely(expr)  __builtin_expect(!!(expr), 0)
#define likely(expr)    __builtin_expect(!!(expr), 1)

void EbCtrbInBase::_pairUp(TebContributor& ctrb,
                           unsigned        idx,
                           const Dgram*    result)
{
  BatchFifo& pending = ctrb.pending();
  if (!pending.empty())
  {
    const Batch* inputs = pending.front();
    if (inputs->index() != idx)
    {
      Batch* batch = ctrb.batch(idx);

      if (unlikely(batch->result()))
      {
        uint64_t pid       = result->seq.pulseId().value();
        uint64_t cachedPid = batch->result()->seq.pulseId().value();
        fprintf(stderr, "%s:\n  Slot occupied by unhandled result: "
                "idx %08x, batch pid %014lx, cached pid %014lx, result pid %014lx\n",
                __PRETTY_FUNCTION__, idx, batch->id(), cachedPid, pid);
        abort();
      }

      batch->result(result);            // Batch is possibly not allocated yet

      result = inputs->result();        // Go on to proccess whatever is ready
    }

    while (result)
    {
      uint64_t iPid = inputs->id();
      uint64_t rPid = result->seq.pulseId().value();
      if (unlikely((iPid ^ rPid) & ~(BATCH_DURATION - 1))) // Include bits above index()
      {
        fprintf(stderr, "%s:\n  Result and Input batches don't match: "
                "Input pid %014lx, Result pid %014lx\n",
                __PRETTY_FUNCTION__, iPid, rPid);
        abort();
      }

      _process(ctrb, result, inputs);

      pending.pop();
      ctrb.release(inputs);      // Release input, and by proxy, result batches
      if (pending.empty())  break;

      inputs = pending.front();
      result = inputs->result();
    }
  }
  else
  {
    Batch* batch = ctrb.batch(idx);

    if (unlikely(batch->result()))
    {
      uint64_t pid       = result->seq.pulseId().value();
      uint64_t cachedPid = batch->result()->seq.pulseId().value();
      fprintf(stderr, "%s:\n  Empty pending FIFO, but slot occupied by unhandled result: "
              "idx %08x, batch pid %014lx, cached pid %014lx, result pid %014lx\n",
              __PRETTY_FUNCTION__, idx, batch->id(), cachedPid, pid);
      abort();
    }

    batch->result(result);              // Batch is possibly not allocated yet
  }
}

void EbCtrbInBase::_process(TebContributor& ctrb,
                            const Dgram*    results,
                            const Batch*    inputs)
{
  const Dgram* result = results;
  const Dgram* input  = static_cast<const Dgram*>(inputs->buffer());
  uint64_t     rPid   = result->seq.pulseId().value();
  uint64_t     iPid   = input->seq.pulseId().value();
  unsigned     rCnt   = 0;
  unsigned     iCnt   = 0;
  while (true)
  {
    // Ignore results for which there is no input
    // This can happen due to this Ctrb being in a different readout group than
    // the one for which the result is for, or having missed a contribution that
    // was subsequently fixed up by the TEB.  In both cases there is no
    // input corresponding to the result.

    if (_prms.verbose)
    {
      unsigned    idx    = inputs->index();
      unsigned    env    = result->env;
      unsigned    src    = result->xtc.src.value();
      unsigned    ctl    = result->seq.pulseId().control();
      const char* svc    = TransitionId::name(result->seq.service());
      size_t      extent = sizeof(*result) + result->xtc.sizeofPayload();
      printf("CtrbIn  found  [%5d]  %15s    @ "
             "%16p, ctl %02x, pid %014lx, sz %4zd, TEB %2d, env %08x, deliver %d [%014lx]\n",
             idx, svc, result, ctl, rPid, extent, src, env, rPid == iPid, iPid);
    }

    if (rPid == iPid)
    {
      //static unsigned lastIndex = 0;
      //unsigned index = reinterpret_cast<uint32_t*>(input->xtc.payload())[1];
      //if (index != ((lastIndex + 1) & (131072 - 1))) {
      //  printf("EbCtrbInBase:  index %u  last %u  diff %d\n", index, lastIndex, index - lastIndex);
      //  printf("EbCtrbInBase:  pid %014lx, env %08x %08x\n", rPid, input->env, result->env);
      //
      //  unsigned     cnt = 0;
      //  const Dgram* res = results;
      //  const Dgram* inp = static_cast<const Dgram*>(inputs->buffer());
      //  while (true)
      //  {
      //    printf("Res: %2d pid = %014lx, env = %08x\n", cnt++, res->seq.pulseId().value(), res->env);
      //
      //    res = reinterpret_cast<const Dgram*>(res->xtc.next());
      //
      //    uint64_t pid = res->seq.pulseId().value();
      //    if ((cnt == MAX_ENTRIES) || !pid)  break; // Handle full list faster
      //  }
      //  cnt = 0;
      //  while (true)
      //  {
      //    printf("Inp: %2d pid = %014lx, env = %08x, index %u\n", cnt++, inp->seq.pulseId().value(), inp->env, reinterpret_cast<uint32_t*>(inp->xtc.payload())[1]);
      //
      //    inp = reinterpret_cast<const Dgram*>(inp->xtc.next());
      //
      //    uint64_t pid = inp->seq.pulseId().value();
      //    if ((cnt == MAX_ENTRIES) || !pid)  break; // Handle full list faster
      //  }
      //}
      //lastIndex = index;

      process(result, inputs->retrieve(iPid));

      ++_eventCount;                  // Don't count events not meant for us

      input = reinterpret_cast<const Dgram*>(input->xtc.next());

      iPid = input->seq.pulseId().value();
      if ((++iCnt == MAX_ENTRIES) || !iPid)  break; // Handle full list faster
    }

    result = reinterpret_cast<const Dgram*>(result->xtc.next());

    rPid = result->seq.pulseId().value();
    if ((++rCnt == MAX_ENTRIES) || !rPid)  break; // Handle full list faster
  }

  if ((iCnt != MAX_ENTRIES) && iPid)
  {
    fprintf(stderr, "%s:\n  Warning: Not all inputs received results, inp: %d %014lx res: %d %014lx\n",
            __PRETTY_FUNCTION__, iCnt, iPid, rCnt, rPid);
  }
}
