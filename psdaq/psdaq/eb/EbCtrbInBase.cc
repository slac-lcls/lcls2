#include "EbCtrbInBase.hh"

#include "Endpoint.hh"
#include "EbLfServer.hh"
#include "Batch.hh"
#include "TebContributor.hh"
#include "StatsMonitor.hh"

#include "utilities.hh"

#include "xtcdata/xtc/Dgram.hh"

#ifdef NDEBUG
//#undef NDEBUG
#endif

#include <cassert>
#include <string>
#include <bitset>


using namespace XtcData;
using namespace Pds::Eb;


EbCtrbInBase::EbCtrbInBase(const TebCtrbParams& prms, StatsMonitor& smon) :
  _transport   (prms.verbose),
  _links       (),
  _maxBatchSize(0),
  _batchCount  (0),
  _eventCount  (0),
  _prms        (prms),
  _region      (nullptr)
{
  smon.metric("TCtbI_RxPdg", _transport.pending(), StatsMonitor::SCALAR);
  smon.metric("TCtbI_BatCt", _batchCount,          StatsMonitor::SCALAR);
  smon.metric("TCtbI_EvtCt", _eventCount,          StatsMonitor::SCALAR);
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

  // Pend for a result datagram (batch) and process it.
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
    printf("CtrbIn  rcvd        %6ld result  [%4d] @ "
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
      uint64_t pid = result->seq.pulseId().value();
      if (unlikely((inputs->id() ^ pid) & ~(BATCH_DURATION - 1))) // Include bits above index()
      {
        fprintf(stderr, "%s:\n  Result doesn't match input batch: "
                "inputs pid %014lx, result pid %014lx\n",
                __PRETTY_FUNCTION__, inputs->id(), pid);
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
  while (true)
  {
    // Process events from our readout group and ignore those of others'
    if (result->readoutGroups() & _prms.readoutGroup)
    {
      // Ignore results for which there are no inputs
      // This can happen due to this Ctrb having missed a contribution that
      // was subsequently fixed up by the TEB, in which case there is no
      // input corresponding to the current result.
      uint64_t rPid = result->seq.pulseId().value();
      uint64_t iPid = input->seq.pulseId().value();
      if (rPid == iPid)
      {
        process(result, inputs->retrieve(rPid));

        ++_eventCount;                  // Don't count events not meant for us
      }
    }

    // Last event in a batch does not have the Batch bit set
    if (!result->seq.isBatch())  break;

    result = reinterpret_cast<const Dgram*>(result->xtc.next());
    input  = reinterpret_cast<const Dgram*>(input->xtc.next());
  }
}
