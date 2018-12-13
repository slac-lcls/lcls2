#include "psdaq/eb/TebContributor.hh"

#include "psdaq/eb/Endpoint.hh"
#include "psdaq/eb/EbLfClient.hh"
#include "psdaq/eb/Batch.hh"
#include "psdaq/eb/EbCtrbInBase.hh"

#include "psdaq/eb/utilities.hh"

#include "xtcdata/xtc/Dgram.hh"

#include <string.h>
#include <cassert>
#include <cstdint>
#include <bitset>
#include <chrono>
#include <string>
#include <thread>

using namespace XtcData;
using namespace Pds::Eb;


TebContributor::TebContributor(const TebCtrbParams& prms) :
  BatchManager (prms.duration, prms.maxBatches, prms.maxEntries, prms.maxInputSize),
  _transport   (new EbLfClient()),
  _links       (),
  _idx2Id      (new unsigned[prms.addrs.size()]),
  _id          (prms.id),
  _numEbs      (std::bitset<64>(prms.builders).count()),
  _batchCount  (0),
  _inFlightOcc (0),
  _inFlightHist(__builtin_ctz(prms.maxBatches), 1.0),
  _depTimeHist (12, double(1 << 16)/1000.),
  _postTimeHist(12, 1.0),
  _postCallHist(12, 1.0),
  _postPrevTime(std::chrono::steady_clock::now()),
  _running     (true),
  _rcvrThread  (nullptr),
  _prms        (prms)
{
  size_t size   = batchRegionSize();    // No need to add Tr space size here
  void*  region = batchRegion();        // Local space for Trs is in the batch region

  for (unsigned i = 0; i < prms.addrs.size(); ++i)
  {
    const char*    addr = prms.addrs[i].c_str();
    const char*    port = prms.ports[i].c_str();
    EbLfLink*      link;
    const unsigned tmo(120000);         // Milliseconds
    if (_transport->connect(addr, port, tmo, &link))
    {
      fprintf(stderr, "%s: Error connecting to EbLfServer at %s:%s\n",
              __func__, addr, port);
      abort();
    }
    if (link->preparePoster(region, size, i, prms.id, prms.verbose))
    {
      fprintf(stderr, "%s: Failed to prepare link to %s:%s\n",
              __func__, addr, port);
      abort();
    }
    _links[link->id()] = link;
    _idx2Id[i] = link->id();

    printf("%s: EbLfServer ID %d connected\n", __func__, link->id());
  }
}

TebContributor::~TebContributor()
{
  if (_idx2Id)     delete [] _idx2Id;
  if (_transport)  delete _transport;
}

void TebContributor::startup(EbCtrbInBase& in)
{
  _rcvrThread = new std::thread([&] { _receiver(in); });
}

void TebContributor::shutdown()
{
  _running = false;

  if (_rcvrThread)  _rcvrThread->join();

  BatchManager::dump();

  char fs[80];
  sprintf(fs, "inFlightOcc_%d.hist", _id);
  printf("Dumped in-flight occupancy histogram to ./%s\n", fs);
  _inFlightHist.dump(fs);

  sprintf(fs, "depTime_%d.hist", _id);
  printf("Dumped departure time histogram to ./%s\n", fs);
  _depTimeHist.dump(fs);

  sprintf(fs, "postTime_%d.hist", _id);
  printf("Dumped post time histogram to ./%s\n", fs);
  _postTimeHist.dump(fs);

  sprintf(fs, "postCallRate_%d.hist", _id);
  printf("Dumped post call rate histogram to ./%s\n", fs);
  _postCallHist.dump(fs);

  for (EbLfLinkMap::iterator it  = _links.begin();
                             it != _links.end(); ++it)
  {
    _transport->shutdown(it->second);
  }
  _links.clear();
}

bool TebContributor::process(const Dgram* datagram, const void* appPrm)
{
  Batch* batch = locate(datagram->seq.pulseId().value()); // Might call post(batch), below
  if (batch)                            // Timed out if nullptr
  {
    Dgram* bdg = batch->buffer(appPrm);

    // Copy entire datagram into the batch (copy ctor doesn't copy payload)
    memcpy(bdg, datagram, sizeof(*datagram) + datagram->xtc.sizeofPayload());

    if (!datagram->seq.isEvent())
    {
      post(batch);
      post(bdg);
      flush();
    }
  }
  return batch;
}

void TebContributor::post(const Batch* batch)
{
  uint32_t    idx    = batch->index();
  unsigned    dst    = _idx2Id[idx % _numEbs];
  EbLfLink*   link   = _links[dst];
  uint32_t    data   = ImmData::value(ImmData::Buffer | ImmData::Response, _id, idx);
  size_t      extent = batch->extent();
  unsigned    offset = idx * maxBatchSize();
  const void* buffer = batch->batch();

  if (_prms.verbose)
  {
    uint64_t pid    = batch->id();
    void*    rmtAdx = (void*)link->rmtAdx(offset);
    printf("CtrbOut posts %9ld    batch[%4d]    @ "
           "%16p, pid %014lx, sz %4zd to   Teb %2d @ %16p, data %08x (%2d entries)\n",
           _batchCount, idx, buffer, pid, extent, link->id(), rmtAdx, data, batch->entries());
  }

  auto t0(std::chrono::steady_clock::now());
  if (link->post(buffer, extent, offset, data) < 0)  return;
  auto t1(std::chrono::steady_clock::now());

  ++_batchCount;

  _updateHists(t0, t1, static_cast<const Dgram*>(buffer)->seq.stamp());

  _inFlightOcc += 1;
  _inFlightHist.bump(_inFlightOcc);
}

void TebContributor::post(const Dgram* nonEvent)
{
  // Non-events are sent to all EBs, except the one that got the batch
  // containing it.  These don't receive responses
  uint64_t pid    = nonEvent->seq.pulseId().value();
  uint32_t idx    = batchId(pid) & (_prms.maxBatches - 1);
  unsigned dst    = _idx2Id[idx % _numEbs];
  unsigned tr     = nonEvent->seq.service();
  uint32_t data   = ImmData::value(ImmData::Transition | ImmData::NoResponse, _id, tr);
  size_t   extent = sizeof(*nonEvent) + nonEvent->xtc.sizeofPayload();
  unsigned offset = batchRegionSize() + tr * _prms.maxInputSize;

  for (EbLfLinkMap::iterator it  = _links.begin();
                             it != _links.end(); ++it)
  {
    EbLfLink* link = it->second;
    if (link->id() != dst)        // Batch posted above included this non-event
    {
      if (_prms.verbose)
      {
        void* rmtAdx = (void*)link->rmtAdx(offset);
        printf("CtrbOut posts          non-event          @ "
               "%16p, pid %014lx, sz %4zd to   Teb %2d @ %16p, data %08x (svc %15s)\n",
               nonEvent, pid, extent, link->id(), rmtAdx, data,
               TransitionId::name(nonEvent->seq.service()));
      }

      link->post(nonEvent, extent, offset, data); // Not a batch
    }
  }
}

void TebContributor::_updateHists(TimePoint_t      t0,
                                 TimePoint_t      t1,
                                 const TimeStamp& stamp)
{
  auto        d  = std::chrono::seconds     { stamp.seconds()     } +
                   std::chrono::nanoseconds { stamp.nanoseconds() };
  TimePoint_t tp { std::chrono::duration_cast<Duration_t>(d) };
  int64_t     dT ( std::chrono::duration_cast<ns_t>(t0 - tp).count() );
  _depTimeHist.bump(dT >> 16);

  dT = std::chrono::duration_cast<us_t>(t1 - t0).count();
  //if (dT > 4095)  printf("postTime = %ld us\n", dT);
  _postTimeHist.bump(dT);
  _postCallHist.bump(std::chrono::duration_cast<us_t>(t0 - _postPrevTime).count());
  _postPrevTime = t0;
}

void TebContributor::_receiver(EbCtrbInBase& in)
{
  pinThread(pthread_self(), _prms.core[1]);

  while (_running)
  {
    if (in.process(*this) < 0)  continue;

    _inFlightOcc -= 1;
  }

  printf("\nShutting down the inbound side...\n");
  in.shutdown();
}

