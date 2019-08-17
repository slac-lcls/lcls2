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
#include <atomic>
#include <thread>

using namespace XtcData;
using namespace Pds;
using namespace Pds::Fabrics;
using namespace Pds::Eb;


EbAppBase::EbAppBase(const EbParams& prms,
                     const uint64_t  duration,
                     const unsigned  maxEntries,
                     const unsigned  maxBuffers) :
  EventBuilder (maxBuffers + TransitionId::NumberOf,
                maxEntries,
                8 * sizeof(prms.contributors), //Revisit: std::bitset<64>(prms.contributors).count(),
                duration,
                prms.verbose),
  _transport   (prms.verbose),
  _maxEntries  (maxEntries),
  _maxBuffers  (maxBuffers),
  //_dummy       (Level::Fragment),
  _verbose     (prms.verbose),
  _bufferCnt   (0),
  _fixupCnt    (0),
  _region      (nullptr),
  _contributors(0),
  _id          (-1)
{
}

int EbAppBase::connect(const EbParams& prms)
{
  int rc;
  unsigned nCtrbs = std::bitset<64>(prms.contributors).count();

  _links.resize(nCtrbs);
  _trRegSize.resize(nCtrbs);
  _maxTrSize.resize(nCtrbs);
  _maxBufSize.resize(nCtrbs);
  _id           = prms.id;
  _contributors = prms.contributors;
  _contract     = prms.contractors;
  _bufferCnt    = 0;
  _fixupCnt     = 0;

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
    _maxTrSize[link->id()]  = prms.maxTrSize[link->id()];
    _trRegSize[link->id()]  = roundUpSize(TransitionId::NumberOf * _maxTrSize[link->id()]);
    _maxBufSize[link->id()] = regSize / _maxBuffers;
    regSize                += _trRegSize[link->id()];  // Ctrbs don't have a transition space
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
    if ( (rc = _links[id]->postCompRecv()) )
    {
      fprintf(stderr, "%s:\n  Failed to post CQ buffers: %d\n",
              __PRETTY_FUNCTION__, rc);
    }

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
  _links.clear();
  _transport.shutdown();

  if (_region)  free(_region);
  _region = nullptr;

  _trRegSize.clear();
  _maxTrSize.clear();
  _maxBufSize.clear();
  _contributors = 0;
  _id           = -1;
  _contract.fill(0);
}

int EbAppBase::process()
{
  int rc;

  // Pend for an input datagram and pass it to the event builder
  uint64_t  data;
  const int tmo = 100;       // milliseconds - Also see EbEvent.cc::MaxTimeouts
  if ( (rc = _transport.pend(&data, tmo)) < 0)
  {
    if (rc == -FI_ETIMEDOUT)  EventBuilder::expired();
    return rc;
  }

  ++_bufferCnt;

  unsigned     flg = ImmData::flg(data);
  unsigned     src = ImmData::src(data);
  unsigned     idx = ImmData::idx(data);
  EbLfLink*    lnk = _links[src];
  size_t       ofs = (ImmData::buf(flg) == ImmData::Buffer)
                   ? (_trRegSize[src] + idx * _maxBufSize[src])
                   : (idx * _maxTrSize[src]);
  const Dgram* idg = static_cast<Dgram*>(lnk->lclAdx(ofs));
  if ( (rc = lnk->postCompRecv()) )
  {
    fprintf(stderr, "%s:\n  Failed to post CQ buffers: %d\n",
            __PRETTY_FUNCTION__, rc);
  }

  if (_verbose)
  {
    unsigned    env = idg->env;
    uint64_t    pid = idg->seq.pulseId().value();
    unsigned    ctl = idg->seq.pulseId().control();
    const char* knd = (ImmData::buf(flg) == ImmData::Buffer)
                    ? "buffer"
                    : TransitionId::name(idg->seq.service());
    printf("EbAp rcvd %6ld %15s[%5d]   @ "
           "%16p, ctl %02x, pid %014lx,            src %2d, env %08x, data %08lx, ext %4d\n",
           _bufferCnt, knd, idx, idg, ctl, pid, lnk->id(), env, data, idg->xtc.extent);
  }

  // Tr space bufSize value is irrelevant since maxEntries will be 1 for that case
  unsigned maxEntries = (ImmData::buf(flg) == ImmData::Buffer) ? _maxEntries : 1;
  size_t   bufSize    = _maxBufSize[src] / maxEntries;

  EventBuilder::process(idg, bufSize, maxEntries, data);

  return 0;
}

void EbAppBase::trim(unsigned dst)
{
  for (unsigned group = 0; group < _contract.size(); ++group)
  {
    _contract[group]  &= ~(1 << dst);
    //_receivers[group] &= ~(1 << dst);
  }
}

uint64_t EbAppBase::contract(const Dgram* ctrb) const
{
  // This method is called when the event is created, which happens when the event
  // builder recognizes the first contribution.  This contribution contains
  // information from the L1 trigger that identifies which readout groups are
  // involved.  This routine can thus look up the expected list of contributors
  // (the contract) to the event for each of the readout groups and logically OR
  // them together to provide the overall contract.  The list of contributors
  // participating in each readout group is provided at configuration time.

  uint64_t contract = 0;
  unsigned groups   = ctrb->readoutGroups();

  while (groups)
  {
    unsigned group = __builtin_ffs(groups) - 1;
    groups &= ~(1 << group);

    contract |= _contract[group];
  }
  return contract;
}

void EbAppBase::fixup(EbEvent* event, unsigned srcId)
{
  ++_fixupCnt;

  if (_verbose)
  {
    fprintf(stderr, "%s:\n  Fixup event %014lx, size %zu, for source %d\n",
            __PRETTY_FUNCTION__, event->sequence(), event->size(), srcId);
  }

  event->damageInc(Damage::DroppedContribution);
}
