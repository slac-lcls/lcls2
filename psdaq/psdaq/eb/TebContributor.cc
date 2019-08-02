#include "TebContributor.hh"

#include "Endpoint.hh"
#include "EbLfClient.hh"
#include "Batch.hh"
#include "EbCtrbInBase.hh"

#include "utilities.hh"

#include "psdaq/service/MetricExporter.hh"
#include "xtcdata/xtc/Dgram.hh"

#ifdef NDEBUG
//#undef NDEBUG
#endif

#include <cassert>
#include <string.h>
#include <cassert>
#include <cstdint>
#include <bitset>
#include <string>
#include <thread>

using namespace XtcData;
using namespace Pds::Eb;


TebContributor::TebContributor(const TebCtrbParams&            prms,
                               std::shared_ptr<MetricExporter>& exporter) :
  _prms        (prms),
  _batMan      (prms.maxInputSize),
  _transport   (prms.verbose),
  _links       (),
  _id          (-1),
  _numEbs      (0),
  _pending     (MAX_BATCHES),
  _batchBase   (roundUpSize(TransitionId::NumberOf * prms.maxInputSize)),
  _batch       (nullptr),
  _eventCount  (0),
  _batchCount  (0)
{
  std::map<std::string, std::string> labels{{"partition", std::to_string(prms.partition)}};
  exporter->add("TCtbO_EvtRt",  labels, MetricType::Rate,    [&](){ return _eventCount;             });
  exporter->add("TCtbO_EvtCt",  labels, MetricType::Counter, [&](){ return _eventCount;             });
  exporter->add("TCtbO_BtAlCt", labels, MetricType::Counter, [&](){ return _batMan.batchAllocCnt(); });
  exporter->add("TCtbO_BtFrCt", labels, MetricType::Counter, [&](){ return _batMan.batchFreeCnt();  });
  exporter->add("TCtbO_BtWtg",  labels, MetricType::Gauge,   [&](){ return _batMan.batchWaiting();  });
  exporter->add("TCtbO_BatCt",  labels, MetricType::Counter, [&](){ return _batchCount;             });
  exporter->add("TCtbO_TxPdg",  labels, MetricType::Gauge,   [&](){ return _transport.pending();    });
  exporter->add("TCtbO_InFlt",  labels, MetricType::Gauge,   [&](){ return _pending.count();        });
}

int TebContributor::connect(const TebCtrbParams& prms)
{
  _id      = prms.id;
  _numEbs  = std::bitset<64>(prms.builders).count();
  _links.resize(prms.addrs.size());
  _pending.clear();

  int    rc;
  void*  region  = _batMan.batchRegion();     // Local space for Trs is in the batch region
  size_t regSize = _batMan.batchRegionSize(); // No need to add Tr space size here

  for (unsigned i = 0; i < prms.addrs.size(); ++i)
  {
    const char*    addr = prms.addrs[i].c_str();
    const char*    port = prms.ports[i].c_str();
    EbLfLink*      link;
    const unsigned tmo(120000);         // Milliseconds
    if ( (rc = _transport.connect(addr, port, tmo, &link)) )
    {
      fprintf(stderr, "%s:\n  Error connecting to TEB at %s:%s\n",
              __PRETTY_FUNCTION__, addr, port);
      return rc;
    }
    if ( (rc = link->preparePoster(prms.id, region, regSize)) )
    {
      fprintf(stderr, "%s:\n  Failed to prepare link with TEB at %s:%s\n",
              __PRETTY_FUNCTION__, addr, port);
      return rc;
    }
    _links[link->id()] = link;

    printf("Outbound link with TEB ID %d connected\n", link->id());
  }

  return 0;
}

void TebContributor::startup(EbCtrbInBase& in)
{
  _batch      = nullptr;
  _eventCount = 0;
  _batchCount = 0;
  _running.store(true, std::memory_order_release);
  _rcvrThread = std::thread([&] { in.receiver(*this, _running); });
}

// Called from another thread to trigger shutting down
void TebContributor::stop()
{
  _running.store(false, std::memory_order_release);

  _batMan.stop();
}

// Called from the current thread to shut it down
void TebContributor::shutdown()
{
  for (auto it = _links.begin(); it != _links.end(); ++it)
  {
    _transport.shutdown(*it);
  }
  _links.clear();

  if (_rcvrThread.joinable())  _rcvrThread.join();

  _batMan.dump();
  _batMan.shutdown();

  _id = -1;
}

void* TebContributor::allocate(const Transition* hdr, const void* appPrm)
{
  if (_prms.verbose > 1)
  {
    const char* svc = TransitionId::name(hdr->seq.service());
    unsigned    ctl = hdr->seq.pulseId().control();
    uint64_t    pid = hdr->seq.pulseId().value();
    unsigned    env = hdr->env;
    printf("Batching  %15s  dg              @ "
           "%16p, ctl %02x, pid %014lx,                    env %08x, prm %p\n",
           svc, hdr, ctl, pid, env, appPrm);
  }

  auto batch = _batMan.allocate(*hdr);
  if (batch)
  {
    ++_eventCount;                      // Count all events handled

    uint64_t pid = hdr->seq.pulseId().value();
    batch->store(pid, appPrm);          // Save the appPrm for _every_ event

    return batch->allocate();
  }
  return batch;                         // Null when terminating
}

void TebContributor::process(const Dgram* datagram)
{
  const auto pid   = datagram->seq.pulseId().value();
  const auto idx   = Batch::batchNum(pid);
  auto       cur   = _batMan.batch(idx);
  bool       flush = !(datagram->seq.isEvent() || (datagram->seq.service() == TransitionId::SlowUpdate));

  if ((_batch && _batch->expired(pid)) || flush)
  {
    if (_batch)  post(_batch);

    if (flush && (_batch != cur))
    {
      post(cur);
      cur = nullptr;
    }

    _batch = cur;
  }
  else if (!_batch)  _batch = cur;

  if (!datagram->seq.isEvent())  post(datagram);
}

void TebContributor::post(const Batch* batch)
{
  _pending.push(batch); // Added to the list only when complete, even if empty

  if (!batch->empty())
  {
    uint32_t    idx    = batch->index();
    unsigned    dst    = idx % _numEbs;
    EbLfLink*   link   = _links[dst];
    uint32_t    data   = ImmData::value(ImmData::Buffer | ImmData::Response, _id, idx);
    size_t      extent = batch->terminate();
    unsigned    offset = _batchBase + idx * _batMan.maxBatchSize();
    const void* buffer = batch->buffer();

    if (_prms.verbose)
    {
      uint64_t pid    = batch->id();
      void*    rmtAdx = (void*)link->rmtAdx(offset);
      printf("CtrbOut posts %9ld    batch[%5d]    @ "
             "%16p,         pid %014lx, sz %6zd, TEB %2d @ %16p, data %08x\n",
             _batchCount, idx, buffer, pid, extent, dst, rmtAdx, data);
    }

    if (link->post(buffer, extent, offset, data) < 0)  return;
  }

  ++_batchCount;                        // Count all batches handled
}

void TebContributor::post(const Dgram* nonEvent)
{
  // Non-events are sent to all EBs, except the one that got the batch
  // containing it.  These EBs won't generate responses.

  uint64_t pid    = nonEvent->seq.pulseId().value();
  uint32_t idx    = Batch::batchNum(pid);
  unsigned dst    = idx % _numEbs;
  unsigned tr     = nonEvent->seq.service();
  uint32_t data   = ImmData::value(ImmData::Transition | ImmData::NoResponse, _id, tr);
  size_t   extent = sizeof(*nonEvent) + nonEvent->xtc.sizeofPayload();
  unsigned offset = tr * _prms.maxInputSize;

  for (auto it = _links.begin(); it != _links.end(); ++it)
  {
    EbLfLink* link = *it;
    if (link->id() != dst)        // Batch posted above included this non-event
    {
      if (_prms.verbose)
      {
        unsigned    env    = nonEvent->env;
        unsigned    ctl    = nonEvent->seq.pulseId().control();
        const char* svc    = TransitionId::name(nonEvent->seq.service());
        void*       rmtAdx = (void*)link->rmtAdx(offset);
        printf("CtrbOut posts    %15s           @ "
               "%16p, ctl %02x, pid %014lx, sz %6zd, TEB %2d, env %08x @ %16p, data %08x\n",
               svc, nonEvent, ctl, pid, extent, link->id(), env, rmtAdx, data);
      }

      link->post(nonEvent, extent, offset, data); // Not a batch
    }
  }
}
