#include "psdaq/eb/EbContributor.hh"

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
#include <unordered_map>

using namespace XtcData;
using namespace Pds::Eb;


EbContributor::EbContributor(const EbCtrbParams& prms) :
  BatchManager (prms.duration, prms.maxBatches, prms.maxEntries, prms.maxInputSize),
  _transport   (new EbLfClient()),
  _links       (), //prms.addrs.size()),
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
  size_t size   = batchRegionSize();
  void*  region = batchRegion();

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

EbContributor::~EbContributor()
{
  if (_idx2Id)     delete [] _idx2Id;
  if (_transport)  delete _transport;
}

void EbContributor::startup(EbCtrbInBase& in)
{
  _rcvrThread = new std::thread([&] { _receiver(in); });
}

void EbContributor::shutdown()
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

bool EbContributor::process(const Dgram* datagram, const void* appPrm)
{
  Batch* batch = allocate(datagram);    // Might call post(batch), below
  if (batch)                            // Timed out if nullptr
  {
    size_t size   = sizeof(*datagram) + datagram->xtc.sizeofPayload();
    void*  buffer = batch->allocate(size, appPrm);

    memcpy(buffer, datagram, size);

    if (!datagram->seq.isEvent())
    {
      post(batch);
      post((const Dgram*)buffer);
      flush();
    }
  }
  return batch;
}

void EbContributor::post(const Batch* batch)
{
  unsigned     dst    = _idx2Id[batchId(batch->id()) % _numEbs];
  EbLfLink*    link   = _links[dst];
  uint32_t     idx    = batch->index();
  uint32_t     data   = ImmData::buffer(_id /*link->index()*/, idx);
  size_t       extent = batch->extent();
  unsigned     offset = idx * maxBatchSize();
  const Dgram* bdg    = batch->datagram();

  // Revisit: Commented out alternate method, EbAppBase::process(), L3EbApp::process(event)
  // Revisit: This doesn't work.  If fdg is the nonEvent, the first flag will make appear to be a batch
  //const Dgram* fdg = (Dgram*)bdg->xtc.payload(); // First Dgram in batch
  //const_cast<Dgram*>(fdg)->seq.first();          // Tag to indicate src.phy is valid
  //const_cast<Dgram*>(fdg)->xtc.src.phy(data);    // Store the return address info

  if (_prms.verbose)
  {
    uint64_t pid    = bdg->seq.pulseId().value();
    void*    rmtAdx = (void*)link->rmtAdx(offset);
    printf("CtrbOut posts %6ld       batch[%4d]    @ "
           "%16p, pid %014lx, sz %4zd to   EB %2d @ %16p (%d entries)\n",
           _batchCount, idx, bdg, pid, extent, link->id(), rmtAdx, batch->entries());
  }

  auto t0(std::chrono::steady_clock::now());
  if (link->post(bdg, extent, offset, data) < 0)  return;
  auto t1(std::chrono::steady_clock::now());

  ++_batchCount;

  _updateHists(t0, t1, bdg->seq.stamp());

  _inFlightOcc += 1;
  _inFlightHist.bump(_inFlightOcc);
}

void EbContributor::post(const Dgram* nonEvent)
{
  // Non-events are sent to all EBs, except the one that got the batch
  // containing it.  These don't receive responses
  unsigned dst    = _idx2Id[batchId(nonEvent->seq.pulseId().value()) % _numEbs];
  unsigned tr     = nonEvent->seq.service();
  size_t   size   = sizeof(*nonEvent) + nonEvent->xtc.sizeofPayload();
  unsigned offset = batchRegionSize() + tr * _prms.maxInputSize;
  for (EbLfLinkMap::iterator it  = _links.begin();
                             it != _links.end(); ++it)
  {
    EbLfLink* lnk = it->second;
    if (lnk->id() != dst)         // Batch posted above included this non-event
    {
      if (_prms.verbose)
      {
        printf("CtrbOut posts          non-event          @ "
               "%16p, pid %014lx, sz %4zd to   EB %2d @ %16p (svc %15s)\n",
               nonEvent, nonEvent->seq.pulseId().value(), size, lnk->id(),
               (void*)lnk->rmtAdx(offset), TransitionId::name(nonEvent->seq.service()));
      }

      uint32_t data = ImmData::transition(_id /*lnk->index()*/, tr);

      lnk->post(nonEvent, size, offset, data); // Not a batch
    }
  }
}

void EbContributor::_updateHists(TimePoint_t      t0,
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

void EbContributor::_receiver(EbCtrbInBase& in)
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

