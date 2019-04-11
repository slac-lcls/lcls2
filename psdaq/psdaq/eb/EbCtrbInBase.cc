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
  _eventCount  (0),
  _prms        (prms),
  _region      (nullptr)
{
  smon.registerIt("TCtbI_RxPdg", _transport.pending(), StatsMonitor::SCALAR);
  smon.registerIt("TCtbI_EvtCt", _eventCount,          StatsMonitor::SCALAR);
}

int EbCtrbInBase::connect(const TebCtrbParams& prms)
{
  int rc;

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
      _maxBatchSize = regSize / prms.maxBatches;

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
  const int tmo = 5000;                 // milliseconds
  if ( (rc = _transport.pend(&data, tmo)) < 0)  return rc;

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
    static unsigned cnt = 0;
    unsigned        ctl = bdg->seq.pulseId().control();
    unsigned        env = bdg->env;
    printf("CtrbIn  rcvd        %6d result  [%4d] @ "
           "%16p, ctl %02x, pid %014lx,          src %2d, env %08x, E %d %zd, result %p\n",
           cnt++, idx, bdg, ctl, pid, lnk->id(), env, ctrb.pending().empty(),
           ctrb.pending().count(), ctrb.batch(idx)->result());
  }

  const Dgram* result  = bdg;
  BatchFifo&   pending = ctrb.pending();
  if (!pending.empty())
  {
    const Batch* inputs = pending.front();
    if (inputs->index() != idx)
    {
      Batch* batch = ctrb.batch(idx);

      assert(batch->result() == nullptr);

      batch->result(result);            // Batch is possibly not allocated yet

      result = inputs->result();        // Go on to proccess whatever is ready
    }

    while (result)
    {
      assert(((inputs->id() ^ result->seq.pulseId().value()) &
              ~(BATCH_DURATION - 1)) == 0); // Include bits above index()

      _process(ctrb, result, inputs);

      pending.pop();
      ctrb.release(inputs);             // Release the input batch
      if (pending.empty())  break;

      inputs = pending.front();
      result = inputs->result();
    }
  }
  else
  {
    Batch* batch = ctrb.batch(idx);

    assert(batch->result() == nullptr);

    batch->result(result);              // Batch is possibly not allocated yet
  }
  return 0;
}

void EbCtrbInBase::_process(TebContributor& ctrb,
                            const Dgram*    results,
                            const Batch*    inputs)
{
  const Dgram* result = results;
  while (true)
  {
    if (result->readoutGroups() & _prms.groups)
    {
      uint64_t pid = result->seq.pulseId().value();
      process(result, inputs->retrieve(pid));

      ++_eventCount;                    // Don't count events not meant for us
    }

    // Last event in batch does not have Batch bit set
    if (!result->seq.isBatch())  break;

    result = reinterpret_cast<const Dgram*>(result->xtc.next());
  }
}
