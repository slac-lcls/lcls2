#include "EbAppBase.hh"

#include "Endpoint.hh"
#include "EbEvent.hh"

#include "EbLfServer.hh"

#include "utilities.hh"

#include "psalg/utils/SysLog.hh"
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
#include <climits>
#include <bitset>
#include <atomic>
#include <thread>
#include <chrono>                       // Revisit: Temporary?

using namespace XtcData;
using namespace Pds;
using namespace Pds::Fabrics;
using namespace Pds::Eb;
using logging  = psalg::SysLog;


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
  _tmoEvtCnt   (0),
  _fixupCnt    (0),
  _region      (nullptr),
  _contributors(0),
  _id          (-1)
{
}

int EbAppBase::configure(const std::string&                     pfx,
                         const EbParams&                        prms,
                         const std::shared_ptr<MetricExporter>& exporter)
{
  unsigned nCtrbs = std::bitset<64>(prms.contributors).count();

  _links.resize(nCtrbs);
  _bufRegSize.resize(nCtrbs);
  _maxTrSize.resize(nCtrbs);
  _maxBufSize.resize(nCtrbs);
  _id           = prms.id;
  _contributors = prms.contributors;
  _contract     = prms.contractors;
  _bufferCnt    = 0;
  _tmoEvtCnt    = 0;
  _fixupCnt     = 0;

  std::map<std::string, std::string> labels{{"instrument", prms.instrument},
                                            {"partition", std::to_string(prms.partition)}};
  exporter->add(pfx+"_RxPdg",  labels, MetricType::Gauge,   [&](){ return _transport.pending(); });
  exporter->add(pfx+"_BfInCt", labels, MetricType::Counter, [&](){ return _bufferCnt;           }); // Inbound
  exporter->add(pfx+"_ToEvCt", labels, MetricType::Counter, [&](){ return _tmoEvtCnt;           });
  exporter->add(pfx+"_FxUpCt", labels, MetricType::Counter, [&](){ return _fixupCnt;            });

  _fixupSrc = &exporter->add(pfx+"_FxUpSc", labels, nCtrbs);
  _ctrbSrc  = &exporter->add(pfx+"_CtrbSc", labels, nCtrbs); // Revisit: For testing

  std::vector<size_t> regSizes(nCtrbs);
  size_t              sumSize = 0;

  int rc;
  if ( (rc = _transport.initialize(prms.ifAddr, prms.ebPort, nCtrbs)) )
  {
    logging::error("%s:\n  Failed to initialize EbLfServer on %s:%s\n",
                   __PRETTY_FUNCTION__, prms.ifAddr, prms.ebPort);
    return rc;
  }

  for (unsigned i = 0; i < _links.size(); ++i)
  {
    EbLfSvrLink*   link;
    const unsigned tmo(120000);         // Milliseconds
    if ( (rc = _transport.connect(&link, _id, tmo)) )
    {
      logging::error("%s:\n  Error connecting to a DRP\n",
                     __PRETTY_FUNCTION__);
      return rc;
    }
    unsigned rmtId = link->id();
    _links[rmtId] = link;

    logging::debug("Inbound link with DRP ID %d connected\n", rmtId);

    size_t regSize;
    if ( (rc = link->prepare(&regSize)) )
    {
      logging::error("%s:\n  Failed to prepare link with DRP ID %d\n",
                     __PRETTY_FUNCTION__, rmtId);
      return rc;
    }
    _bufRegSize[rmtId] = regSize;
    _maxBufSize[rmtId] = regSize / (_maxBuffers * _maxEntries);
    _maxTrSize[rmtId]  = prms.maxTrSize[rmtId];
    regSize           += roundUpSize(TransitionId::NumberOf * _maxTrSize[rmtId]);  // Ctrbs don't have a transition space
    regSizes[rmtId]    = regSize;
    sumSize           += regSize;
  }

  _region = allocRegion(sumSize);
  if (!_region)
  {
    logging::error("%s:\n  No memory found for Input MR of size %zd\n",
                   __PRETTY_FUNCTION__, sumSize);
    return ENOMEM;
  }

  // Note that this loop can't be combined with the one above due to the exchange protocol
  char* region = reinterpret_cast<char*>(_region);
  for (unsigned rmtId = 0; rmtId < _links.size(); ++rmtId)
  {
    EbLfSvrLink* link = _links[rmtId];
    if ( (rc = link->setupMr(region, regSizes[rmtId])) )
    {
      logging::error("%s:\n  Failed to set up Input MR for DRP ID %d, "
                     "%p:%p, size %zd\n", __PRETTY_FUNCTION__,
                     rmtId, region, region + regSizes[rmtId], regSizes[rmtId]);
      if (_region)  free(_region);
      _region = nullptr;
      return rc;
    }

    if (link->postCompRecv())
    {
      logging::warning("%s:\n  Failed to post CQ buffers for DRP ID %d\n",
                       __PRETTY_FUNCTION__, rmtId);
    }

    region += regSizes[rmtId];

    logging::info("Inbound link with DRP ID %d connected and configured\n", rmtId);
  }

  return 0;
}

void EbAppBase::shutdown()
{
  EventBuilder::dump(0);
  EventBuilder::clear();

  for (auto it = _links.begin(); it != _links.end(); ++it)
  {
    _transport.disconnect(*it);
  }
  _links.clear();
  _transport.shutdown();

  if (_region)  free(_region);
  _region = nullptr;

  _bufRegSize.clear();
  _maxBufSize.clear();
  _maxTrSize.clear();
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

  unsigned       flg = ImmData::flg(data);
  unsigned       src = ImmData::src(data);
  unsigned       idx = ImmData::idx(data);
  EbLfSvrLink*   lnk = _links[src];
  size_t         ofs = (ImmData::buf(flg) == ImmData::Buffer)
                     ? (                   idx * _maxBufSize[src]) // In batch/buffer region
                     : (_bufRegSize[src] + idx * _maxTrSize[src]); // Tr region for non-selected EB is after batch/buffer region
  const EbDgram* idg = static_cast<EbDgram*>(lnk->lclAdx(ofs));
  if ( (rc = lnk->postCompRecv()) )
  {
    logging::warning("%s:\n  Failed to post CQ buffers for DRP ID %d\n",
                     __PRETTY_FUNCTION__, src);
  }

  _ctrbSrc->observe(double(src));       // Revisit: For testing

  if (unlikely(_verbose >= VL_BATCH))
  {
    unsigned    env = idg->env;
    uint64_t    pid = idg->pulseId();
    unsigned    ctl = idg->control();
    const char* knd = TransitionId::name(idg->service());
    printf("EbAp rcvd %9ld %15s[%8d]   @ "
           "%16p, ctl %02x, pid %014lx, env %08x,            src %2d, data %08lx\n",
           _bufferCnt, knd, idx, idg, ctl, pid, env, lnk->id(), data);
  }

  EventBuilder::process(idg, _maxBufSize[src], data);

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

uint64_t EbAppBase::contract(const EbDgram* ctrb) const
{
  // This method is called when the event is created, which happens when the event
  // builder recognizes the first contribution.  This contribution contains
  // information from the L1 trigger that identifies which readout groups are
  // involved.  This routine can thus look up the expected list of contributors
  // (the contract) to the event for each of the readout groups and logically OR
  // them together to provide the overall contract.  The list of contributors
  // participating in each readout group is provided at configuration time.

  uint64_t contract = 0;
  uint16_t groups   = ctrb->readoutGroups();

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
  if (event->alive())
    ++_fixupCnt;
  else
    ++_tmoEvtCnt;

  _fixupSrc->observe(double(srcId));

  //if (_verbose >= VL_EVENT)
  {
    using ms_t  = std::chrono::milliseconds;   // Revisit: Temporary?
    auto  now   = fast_monotonic_clock::now(); // Revisit: Temporary?
    const EbDgram* dg = event->creator();
    printf("%s %15s %014lx, size %2zu, for source %2d, RoGs %04hx, contract %016lx, remaining %016lx, age %ld\n",
           event->alive() ? "Fixed-up" : "Timed-out",
           TransitionId::name(dg->service()), event->sequence(), event->size(),
           srcId, dg->readoutGroups(), event->contract(), event->remaining(),
           std::chrono::duration_cast<ms_t>(now - event->t0).count());
  }

  event->damage(Damage::DroppedContribution);
}
