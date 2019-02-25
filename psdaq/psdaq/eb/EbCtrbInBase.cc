#include "EbCtrbInBase.hh"

#include "psdaq/eb/Endpoint.hh"
#include "psdaq/eb/EbLfServer.hh"
#include "psdaq/eb/Batch.hh"
#include "psdaq/eb/TebContributor.hh"
#include "psdaq/eb/utilities.hh"

#include "xtcdata/xtc/Dgram.hh"

#include <string>
#include <bitset>


using namespace XtcData;
using namespace Pds::Eb;


EbCtrbInBase::EbCtrbInBase(const TebCtrbParams& prms) :
  _transport   (prms.verbose),
  _links       (),
  _maxBatchSize(0),
  _ebCntHist   ( 6, 1.0),               // Up to 64 possible EBs
  _rttHist     (12, 1.0),
  _pendTimeHist(12, 1.0),
  _pendCallHist(12, 1.0),
  _pendPrevTime(std::chrono::steady_clock::now()),
  _prms        (prms),
  _regions     ()
{
}

int EbCtrbInBase::connect(const TebCtrbParams& prms)
{
  int rc;

  unsigned numEbs = std::bitset<64>(prms.builders).count();
  _links.resize(numEbs);
  _regions.resize(numEbs);

  if ( (rc = _transport.initialize(prms.ifAddr, prms.port, numEbs)) )
  {
    fprintf(stderr, "%s:\n  Failed to initialize EbLfServer\n",
            __PRETTY_FUNCTION__);
    return rc;
  }

  // Since each EB handles a specific batch, one region can be shared by all
  for (unsigned i = 0; i < numEbs; ++i)
  {
    EbLfLink*      link;
    const unsigned tmo(120000);         // Milliseconds
    if ( (rc = _transport.connect(&link, tmo)) )
    {
      fprintf(stderr, "%s: Error connecting to EbLfClient[%d]\n",
              __PRETTY_FUNCTION__, i);
      return rc;
    }
    size_t regSize;
    if ( (rc = link->preparePender(prms.id, &regSize)) )
    {
      fprintf(stderr, "%s: Failed to prepare link[%d]\n",
              __PRETTY_FUNCTION__, i);
      return rc;
    }
    _links[link->id()] = link;

    _regions[i] = allocRegion(regSize);
    if (!_regions[i])
    {
      fprintf(stderr, "%s: No memory found for region %d of size %zd\n",
              __PRETTY_FUNCTION__, i, regSize);
      return ENOMEM;
    }
    if ( (rc = link->setupMr(_regions[i], regSize)) )
    {
      fprintf(stderr, "%s: Failed to set up MemoryRegion %d\n",
              __PRETTY_FUNCTION__, i);
      return rc;
    }
    link->postCompRecv();

    size_t maxBatchSize = regSize / prms.maxBatches;
    if      (_maxBatchSize == 0)  _maxBatchSize = maxBatchSize;
    else if (_maxBatchSize != maxBatchSize)
    {
      fprintf(stderr, "%s: MaxBatchSize (%zd) can't differ between EBs (%zd from Id %d)\n",
              __PRETTY_FUNCTION__, _maxBatchSize, maxBatchSize, link->id());
      return -1;
    }

    printf("Inbound link with TEB ID %d connected\n", link->id());
  }

  return 0;
}

void EbCtrbInBase::shutdown()
{
  char fs[80];
  sprintf(fs, "ebCntHist_%d.hist", _prms.id);
  printf("Dumped EB count histogram to ./%s\n", fs);
  _ebCntHist.dump(fs);

  sprintf(fs, "rtt_%d.hist", _prms.id);
  printf("Dumped RTT histogram to ./%s\n", fs);
  _rttHist.dump(fs);

  sprintf(fs, "pendTime_%d.hist", _prms.id);
  printf("Dumped pend time histogram to ./%s\n", fs);
  _pendTimeHist.dump(fs);

  sprintf(fs, "pendCallRate_%d.hist", _prms.id);
  printf("Dumped pend call rate histogram to ./%s\n", fs);
  _pendCallHist.dump(fs);

  for (auto it = _links.begin(); it != _links.end(); ++it)
  {
    _transport.shutdown(*it);
  }
  _links.clear();
  for (unsigned i = 0; i < _regions.size(); ++i)
  {
    if (_regions[i])  free(_regions[i]);
  }
}

int EbCtrbInBase::process(BatchManager& batMan)
{
  // Pend for a result datagram (batch) and process it.
  uint64_t  data;
  int       rc;
  const int tmo = 5000;                 // milliseconds
  auto t0(std::chrono::steady_clock::now());
  if ( (rc = _transport.pend(&data, tmo)) < 0)  return rc;
  auto t1(std::chrono::steady_clock::now());

  unsigned     src = ImmData::src(data);
  unsigned     idx = ImmData::idx(data);
  EbLfLink*    lnk = _links[src];
  const Dgram* bdg = (const Dgram*)(lnk->lclAdx(idx * _maxBatchSize));
  lnk->postCompRecv();

  if (_prms.verbose)
  {
    static unsigned cnt = 0;
    uint64_t        pid = bdg->seq.pulseId().value();
    unsigned        ctl = bdg->seq.pulseId().control();
    size_t          sz  = sizeof(*bdg) + bdg->xtc.sizeofPayload();
    printf("CtrbIn  rcvd        %6d result  [%4d] @ "
           "%16p, ctl %02x, pid %014lx, sz %4zd, src %2d\n",
           cnt++, idx, bdg, ctl, pid, sz, lnk->id());
  }

  // Makes sense only when t1 and bdg->seq.stamp() have a common clock
  _updateHists(t0, t1, bdg->seq.stamp());
  _ebCntHist.bump(lnk->id());

  Dgram const* result = bdg;
  const Batch* inputs = batMan.batch(idx);
  //printf("        data %08lx           idx %4d rPid %014lx iPid %014lx               id %2d svc %d\n",
  //       data, idx, result->seq.pulseId().value(), inputs->datagram()->seq.pulseId().value(),
  //       lnk->id(), result->seq.service());
  unsigned i = 0;
  while (true)
  {
    process(result, inputs->appParm(i++));

    if (!result->seq.isBatch())  break; // Last event in batch does not have Batch bit set

    result = reinterpret_cast<const Dgram*>(result->xtc.next());
  }

  batMan.release(inputs);

  return 0;
}

void EbCtrbInBase::_updateHists(TimePoint_t      t0,
                                TimePoint_t      t1,
                                const TimeStamp& stamp)
{
  auto        d  = std::chrono::seconds     { stamp.seconds()     } +
                   std::chrono::nanoseconds { stamp.nanoseconds() };
  TimePoint_t tp { std::chrono::duration_cast<Duration_t>(d) };
  int64_t     dT ( std::chrono::duration_cast<us_t>(t1 - tp).count() );
  _rttHist.bump(dT);
  //printf("In  Batch %014lx RTT  = %ld S, %ld ns\n", bdg->seq.pulseId().value(), dS, dN);

  dT = std::chrono::duration_cast<us_t>(t1 - t0).count();
  //if (dT > 4095)  printf("pendTime = %ld us\n", dT);
  _pendTimeHist.bump(dT);
  _pendCallHist.bump(std::chrono::duration_cast<us_t>(t0 - _pendPrevTime).count());
  _pendPrevTime = t0;
}
