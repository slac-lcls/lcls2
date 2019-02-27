#include "EbAppBase.hh"

#include "Endpoint.hh"
#include "EbEvent.hh"

#include "EbLfServer.hh"

#include "utilities.hh"

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


EbAppBase::EbAppBase(const EbParams& prms) :
  EventBuilder (prms.maxBuffers + TransitionId::NumberOf,
                prms.maxEntries,
                64, //std::bitset<64>(prms.contributors).count(),
                prms.duration,
                prms.verbose),
  _defContract (0),
  _contract    ( { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } ),
  _transport   (prms.verbose),
  _links       (),
  _trSize      (roundUpSize(TransitionId::NumberOf * prms.maxTrSize)),
  _maxTrSize   (prms.maxTrSize),
  _maxBufSize  (),
  //_dummy       (Level::Fragment),
  _verbose     (prms.verbose),
  _region      (nullptr),
  _id          (-1)
{
}

int EbAppBase::connect(const EbParams& prms)
{
  int rc;
  unsigned nCtrbs = std::bitset<64>(prms.contributors).count();

  _links.resize(nCtrbs);
  _maxBufSize.resize(nCtrbs);
  _defContract = prms.contributors;
  _id          = prms.id;

  if ( (rc = _transport.initialize(prms.ifAddr, prms.ebPort, nCtrbs)) )
  {
    fprintf(stderr, "%s:\n  Failed to initialize EbLfServer\n",
            __PRETTY_FUNCTION__);
    return rc;
  }

  std::vector<size_t> regSizes(nCtrbs);
  size_t              sumSize = 0;

  for (unsigned i = 0; i < nCtrbs; ++i)
  {
    EbLfLink*      link;
    const unsigned tmo(120000);         // Milliseconds
    if ( (rc = _transport.connect(&link, tmo)) )
    {
      fprintf(stderr, "%s:\n  Error connecting to Ctrb %d\n",
              __PRETTY_FUNCTION__, i);
      return rc;
    }
    size_t regSize;
    if ( (rc = link->preparePender(prms.id, &regSize)) )
    {
      fprintf(stderr, "%s:\n  Failed to prepare link with Ctrb %d\n",
              __PRETTY_FUNCTION__, i);
      return rc;
    }
    _links[link->id()]      = link;
    _maxBufSize[link->id()] = regSize / prms.maxBuffers;
    regSize                += _trSize;  // Ctrbs don't have a transition space
    regSizes[link->id()]    = regSize;
    sumSize                += regSize;
  }

  _region = allocRegion(sumSize);
  if (!_region)
  {
    fprintf(stderr, "%s:\n  No memory found for Input MR of size %zd\n",
            __PRETTY_FUNCTION__, sumSize);
    return ENOMEM;
  }

  char* region = reinterpret_cast<char*>(_region);
  for (unsigned id = 0; id < nCtrbs; ++id)
  {
    if ( (rc = _links[id]->setupMr(region, regSizes[id])) )
    {
      fprintf(stderr, "%s:\n  Failed to set up Input MR for Ctrb ID %d, "
              "%p:%p, size %zd\n", __PRETTY_FUNCTION__,
              id, region, region + regSizes[id], regSizes[id]);
      return rc;
    }
    _links[id]->postCompRecv();

    printf("Inbound link with Ctrb ID %d connected\n", id);

    region += regSizes[id];
  }

  return 0;
}

void EbAppBase::shutdown()
{
  EventBuilder::dump(0);
  EventBuilder::clear();

  for (auto it = _links.begin(); it != _links.end(); ++it)
  {
    _transport.shutdown(*it);
  }

  if (_region)  free(_region);
  _region = nullptr;

  _links.clear();
  _maxBufSize.clear();
  _defContract = 0;
  _id          = -1;
}

int EbAppBase::process()
{
  // Pend for an input datagram and pass it to the event builder
  uint64_t  data;
  int       rc;
  const int tmo = 5000;                 // milliseconds
  auto t0(std::chrono::steady_clock::now());
  if ( (rc = _transport.pend(&data, tmo)) < 0)  return rc;
  auto t1(std::chrono::steady_clock::now());

  unsigned     flg = ImmData::flg(data);
  unsigned     src = ImmData::src(data);
  unsigned     idx = ImmData::idx(data);
  EbLfLink*    lnk = _links[src];
  size_t       ofs = (ImmData::buf(flg) == ImmData::Buffer)
                   ? (_trSize + idx * _maxBufSize[src])
                   : (idx * _maxTrSize);
  const Dgram* idg = static_cast<Dgram*>(lnk->lclAdx(ofs));
  lnk->postCompRecv();

  if (_verbose)
  {
    static unsigned cnt = 0;
    uint64_t        pid = idg->seq.pulseId().value();
    unsigned        ctl = idg->seq.pulseId().control();
    const char*     knd = (ImmData::buf(flg) == ImmData::Buffer)
                        ? "buffer"
                        : TransitionId::name(idg->seq.service());
    printf("EbAp rcvd %6d %15s[%4d]    @ "
           "%16p, ctl %02x, pid %014lx,          src %2d, data %08lx, ext %4d\n",
           cnt++, knd, idx, idg, ctl, pid, lnk->id(), data, idg->xtc.extent);
  }

  EventBuilder::process(idg, data);

  return 0;
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

  if (_verbose)
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
