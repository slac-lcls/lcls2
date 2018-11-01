#include "EbCtrbInBase.hh"

#include "psdaq/eb/Endpoint.hh"
#include "psdaq/eb/EbLfServer.hh"
#include "psdaq/eb/Batch.hh"
#include "psdaq/eb/EbContributor.hh"    // For EbCtrbParams
#include "psdaq/eb/utilities.hh"

#include "xtcdata/xtc/Dgram.hh"

#include <string>
#include <bitset>


using namespace XtcData;
using namespace Pds::Eb;


EbCtrbInBase::EbCtrbInBase(const EbCtrbParams& prms) :
  _numEbs      (std::bitset<64>(prms.builders).count()),
  _maxBatchSize(roundUpSize(sizeof(Dgram) + prms.maxEntries * prms.maxInputSize)),
  _region      (allocRegion(prms.maxBatches * _maxBatchSize)),
  _transport   (new EbLfServer(prms.ifAddr, prms.port.c_str())),
  _links       (), //_numEbs),
  _ebCntHist   ( 6, 1.0),               // Up to 64 possible EBs
  _rttHist     (12, 1.0),
  _pendTimeHist(12, 1.0),
  _pendCallHist(12, 1.0),
  _pendPrevTime(std::chrono::steady_clock::now()),
  _prms        (prms)
{
  _initialize(__func__);
}

EbCtrbInBase::~EbCtrbInBase()
{
  if (_transport)
  {
    for (EbLfLinkMap::iterator it  = _links.begin();
                               it != _links.end(); ++it)
    {
      _transport->shutdown(it->second);
    }
    _links.clear();
    delete _transport;
  }
}

void EbCtrbInBase::_initialize(const char* who)
{
  size_t size = _prms.maxBatches * _maxBatchSize;

  if (_region == nullptr)
  {
    fprintf(stderr, "%s: No memory found for a result region of size %zd\n",
            who, size);
    abort();
  }

  // Since each EB handles a specific batch, one region can be shared by all
  for (unsigned i = 0; i < _numEbs; ++i)
  {
    EbLfLink*      link;
    const unsigned tmo(120000);         // Milliseconds
    if (_transport->connect(&link, tmo))
    {
      fprintf(stderr, "%s: Error connecting to EbLfClient[%d]\n", who, i);
      abort();
    }
    if (link->preparePender((char*)_region, size, i, _prms.id, _prms.verbose))
    {
      fprintf(stderr, "%s: Failed to prepare link[%d]\n", who, i);
      abort();
    }
    _links[link->id()] = link;

    printf("%s: EbLfClient ID %d connected\n", who, link->id());
  }
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
}

int EbCtrbInBase::process(BatchManager& batMan)
{
  // Pend for a result datagram (batch) and process it.
  uint64_t data;
  const int tmo = 5000;                 // milliseconds
  auto t0(std::chrono::steady_clock::now());
  if (_transport->pend(&data, tmo))  return -1;
  auto t1(std::chrono::steady_clock::now());

  unsigned     src = ImmData::src(data);
  unsigned     idx = ImmData::idx(data);
  EbLfLink*    lnk = _links[src];
  const Dgram* bdg = (const Dgram*)(lnk->lclAdx(idx * _maxBatchSize));

  lnk->postCompRecv();

  if (_prms.verbose)
  {
    static unsigned cnt    = 0;
    uint64_t        pid    = bdg->seq.pulseId().value();
    size_t          extent = sizeof(*bdg) + bdg->xtc.sizeofPayload();
    printf("CtrbIn  rcvd        %6d result  [%4d] @ "
           "%16p, pid %014lx, sz %4zd from Teb %2d\n",
           cnt++, idx, bdg, pid, extent, lnk->id());
  }

  // Makes sense only when t1 and bdg->seq.stamp() have a common clock
  _updateHists(t0, t1, bdg->seq.stamp());
  _ebCntHist.bump(lnk->id());

  Dgram const*       result = (Dgram const*)bdg->xtc.payload();
  Dgram const* const last   = (Dgram const*)bdg->xtc.next();
  const Batch*       inputs = batMan.batch(idx);
  //printf("        data %08lx           idx %4d rPid %014lx iPid %014lx               id %2d svc %d\n",
  //       data, idx, result->seq.pulseId().value(), inputs->datagram()->seq.pulseId().value(),
  //       lnk->id(), result->seq.service());
  unsigned i = 0;
  while (result != last)
  {
    process(result, inputs->appParm(i++));

    result = (Dgram const*)result->xtc.next();
  }

  batMan.deallocate(inputs);

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
