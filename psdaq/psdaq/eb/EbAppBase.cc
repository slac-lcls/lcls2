#include "EbAppBase.hh"

#include "Endpoint.hh"
#include "EbEvent.hh"

#include "EbLfServer.hh"

#include "utilities.hh"

#include "psdaq/service/Histogram.hh"
#include "xtcdata/xtc/Dgram.hh"

#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif
#include <sched.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <inttypes.h>
#include <assert.h>
#include <climits>
#include <bitset>
#include <chrono>
#include <atomic>
#include <thread>

using namespace XtcData;
using namespace Pds;
using namespace Pds::Fabrics;
using namespace Pds::Eb;

using Duration_t = std::chrono::steady_clock::duration;
using us_t       = std::chrono::microseconds;
using ns_t       = std::chrono::nanoseconds;

unsigned EbAppBase::lverbose = 0;


EbAppBase::EbAppBase(const char*        ifAddr,
                     const std::string& port,
                     unsigned           id,
                     uint64_t           duration,
                     unsigned           maxBuffers,
                     unsigned           maxEntries,
                     size_t             maxInDgSize,
                     size_t             maxTrDgSize,
                     uint64_t           contributors) :
  EventBuilder (maxBuffers + TransitionId::NumberOf, maxEntries, std::bitset<64>(contributors).count(), duration),
  _maxBufSize  (roundUpSize(maxEntries * maxInDgSize)),
  _region      (allocRegion(std::bitset<64>(contributors).count() *
                            (maxBuffers * _maxBufSize +
                             roundUpSize(TransitionId::NumberOf * maxTrDgSize)))),
  _transport   (new EbLfServer(ifAddr, port.c_str())),
  _links       (),
  _id          (id),
  _defContract (contributors),
  _contract    ( { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } ),
  _trOffset    (TransitionId::NumberOf),
  //_dummy       (Level::Fragment),
  _ctrbCntHist (6, 1.0),                // Up to 64 Ctrbs
  _arrTimeHist (12, double(1 << 16)/1000.),
  _pendTimeHist(12, double(1 <<  8)/1000.),
  _pendCallHist(12, 1.0),
  _pendPrevTime(std::chrono::steady_clock::now())
{
  size_t   size   = maxBuffers * _maxBufSize + roundUpSize(TransitionId::NumberOf * maxTrDgSize);
  unsigned nCtrbs = std::bitset<64>(contributors).count();

  if (_region == nullptr)
  {
    fprintf(stderr, "%s: No memory found for a input region of size %zd\n",
            __func__, nCtrbs * size);
    abort();
  }

  size_t offset = maxBuffers * _maxBufSize;
  for (unsigned tr = 0; tr < _trOffset.size(); ++tr)
  {
    _trOffset[tr] = offset + tr * maxTrDgSize;
  }

  char* region = static_cast<char*>(_region);
  for (unsigned i = 0; i < nCtrbs; ++i)
  {
    EbLfLink*      link;
    const unsigned tmo(120000);         // Milliseconds
    if (_transport->connect(&link, tmo))
    {
      fprintf(stderr, "%s: Error connecting to EbLfClient[%d]\n", __func__, i);
      abort();
    }
    if (link->preparePender(region, size, i, id, lverbose))
    {
      fprintf(stderr, "%s: Failed to prepare link[%d]\n", __func__, i);
      abort();
    }
    _links[link->id()] = link;

    printf("EbLfClient ID %d connected\n", link->id());

    region += size;
  }
}

EbAppBase::~EbAppBase()
{
  if (_transport)  delete _transport;
  if (_region)     free  (_region);
}

void EbAppBase::shutdown()
{
  printf("\nEvent builder dump:\n");
  EventBuilder::dump(0);

  char fs[80];
  sprintf(fs, "ctrbCntHist_%d.hist", _id);
  printf("Dumped contributor count histogram to ./%s\n", fs);
  _ctrbCntHist.dump(fs);

  sprintf(fs, "arrTime_%d.hist", _id);
  printf("Dumped arrival time histogram to ./%s\n", fs);
  _arrTimeHist.dump(fs);

  sprintf(fs, "pendTime_%d.hist", _id);
  printf("Dumped pend time histogram to ./%s\n", fs);
  _pendTimeHist.dump(fs);

  sprintf(fs, "pendCallRate_%d.hist", _id);
  printf("Dumped pend call rate histogram to ./%s\n", fs);
  _pendCallHist.dump(fs);

  for (EbLfLinkMap::iterator it  = _links.begin();
                             it != _links.end(); ++it)
  {
    _transport->shutdown(it->second);
  }
  _links.clear();
}

void EbAppBase::process()
{
  // Pend for an input datagram and pass it to the event builder
  uint64_t  data;
  const int tmo = 5000;                 // milliseconds
  auto t0(std::chrono::steady_clock::now());
  if (_transport->pend(&data, tmo))  return;
  auto t1(std::chrono::steady_clock::now());

  unsigned     flg = ImmData::flg(data);
  unsigned     src = ImmData::src(data);
  unsigned     idx = ImmData::idx(data);
  EbLfLink*    lnk = _links[src];
  size_t       ofs = (ImmData::buf(flg) == ImmData::Buffer) ? idx * _maxBufSize
                                                            : _trOffset[idx];
  const Dgram* idg = static_cast<Dgram*>(lnk->lclAdx(ofs));
  lnk->postCompRecv();

  if (lverbose)
  {
    static unsigned cnt = 0;
    uint64_t        pid = idg->seq.pulseId().value();
    printf("EbAp rcvd %6d %15s[%4d]    @ "
           "%16p, pid %014lx, sz   ?? from Ctrb %2d, data %08lx\n",
           cnt++, (ImmData::buf(flg) == ImmData::Buffer) ? "buffer" : TransitionId::name(idg->seq.service()),
           idx, idg, pid, lnk->id(), data);
  }

  _updateHists(t0, t1, idg->seq.stamp());
  _ctrbCntHist.bump(lnk->id());

  EventBuilder::process(idg, data);     // Comment out for open-loop running
}

void EbAppBase::_updateHists(TimePoint_t      t0,
                             TimePoint_t      t1,
                             const TimeStamp& stamp)
{
  auto        d  = std::chrono::seconds     { stamp.seconds()     } +
                   std::chrono::nanoseconds { stamp.nanoseconds() };
  TimePoint_t tp { std::chrono::duration_cast<Duration_t>(d) };
  int64_t     dT ( std::chrono::duration_cast<ns_t>(t1 - tp).count() );
  _arrTimeHist.bump(dT >> 16);

  dT = std::chrono::duration_cast<us_t>(t1 - t0).count();
  //if (dT > 4095)  printf("pendTime = %ld ns\n", dT);
  _pendTimeHist.bump(dT);
  _pendCallHist.bump(std::chrono::duration_cast<us_t>(t0 - _pendPrevTime).count());
  _pendPrevTime = t0;
}

uint64_t EbAppBase::contract(const Dgram* ctrb) const
{
  // This method called when the event is created, which happens when the event
  // builder recognizes the first contribution.  This contribution contains
  // information from the L1 trigger that identifies which readout groups are
  // involved.  This routine can thus look up the expected list of contributors
  // (the contract) to the event for each of the readout groups and logically OR
  // them together to provide the overall contract.  The list of contributors
  // participating in each readout group is provided at configuration time.

  if (ctrb->seq.isEvent())
  {
    uint64_t contract = 0;
    unsigned groups   = static_cast<const L1Dgram*>(ctrb)->readoutGroups();

    while (groups)
    {
      unsigned group = __builtin_ffs(groups) - 1;
      groups &= ~(1 << group);

      contract |= _contract[group];
    }

    if (contract)  return contract;
  }
  return _defContract;
}

void EbAppBase::fixup(EbEvent* event, unsigned srcId)
{
  // Revisit: Nothing can usefully be done here since there is no way to
  //          know the buffer index to be used for the result, I think

  if (lverbose)
  {
    fprintf(stderr, "%s:\n  Fixup event %014lx, size %zu, for source %d\n",
            __PRETTY_FUNCTION__, event->sequence(), event->size(), srcId);
  }

  // Revisit: What can be done here?
  //          And do we want to send a result to a contributor we haven't heard from?
  //Datagram* datagram = (Datagram*)event->data();
  //Damage    dropped(1 << Damage::DroppedContribution);
  //Src       source(srcId, 0);
  //
  //datagram->xtc.damage.increase(Damage::DroppedContribution);
  //
  //new(&datagram->xtc.tag) Xtc(_dummy, source, dropped); // Revisit: No tag, no TC
}
