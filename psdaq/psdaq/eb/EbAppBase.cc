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
using logging          = psalg::SysLog;
using MetricExporter_t = std::shared_ptr<MetricExporter>;
using ms_t             = std::chrono::milliseconds;


EbAppBase::EbAppBase(const EbParams&         prms,
                     const MetricExporter_t& exporter,
                     const std::string&      pfx,
                     const uint64_t          duration,
                     const unsigned          maxEntries,
                     const unsigned          maxBuffers) :
  EventBuilder (maxBuffers + TransitionId::NumberOf,
                maxEntries,
                MAX_DRPS, //Revisit: std::bitset<64>(prms.contributors).count(),
                duration,
                prms.verbose),
  _transport   (prms.verbose, prms.kwargs),
  _maxEntries  (maxEntries),
  _maxBuffers  (maxBuffers),
  _verbose     (prms.verbose),
  _bufferCnt   (0),
  _contributors(0),
  _id          (-1),
  _exporter    (exporter),
  _pfx         (pfx)
{
  std::map<std::string, std::string> labels{{"instrument", prms.instrument},
                                            {"partition", std::to_string(prms.partition)},
                                            {"detname", prms.alias},
                                            {"alias", prms.alias},
                                            {"eb", pfx}};
  uint64_t depth = (maxBuffers + TransitionId::NumberOf) * maxEntries;
  exporter->constant("EB_EvPlDp", labels, depth);

  exporter->add("EB_EvAlCt", labels, MetricType::Counter, [&](){ return  eventAllocCnt();     });
  exporter->add("EB_EvFrCt", labels, MetricType::Counter, [&](){ return  eventFreeCnt();      });
  exporter->add("EB_RxPdg",  labels, MetricType::Gauge,   [&](){ return _transport.pending(); });
  exporter->add("EB_BfInCt", labels, MetricType::Counter, [&](){ return _bufferCnt;           }); // Inbound
  exporter->add("EB_ToEvCt", labels, MetricType::Counter, [&](){ return  timeoutCnt();        });
  exporter->add("EB_FxUpCt", labels, MetricType::Counter, [&](){ return  fixupCnt();          });
  exporter->add("EB_CbMsMk", labels, MetricType::Gauge,   [&](){ return  missing();           });
}

EbAppBase::~EbAppBase()
{
  for (auto& region : _region)
  {
    if (region)  free(region);
    region = nullptr;
  }
  _region.clear();
}

void EbAppBase::shutdown()
{
  if (_id != unsigned(-1))              // Avoid shutting down if already done
  {
    unconfigure();
    disconnect();

    _transport.shutdown();
  }
}

void EbAppBase::disconnect()
{
  for (auto link : _links)
  {
    _transport.disconnect(link);
  }
  _links.clear();

  _id           = -1;
  _contributors = 0;
  _contract     .fill(0);
  _bufRegSize   .clear();
  _maxBufSize   .clear();
  _maxTrSize    .clear();
}

void EbAppBase::unconfigure()
{
  EventBuilder::dump(0);
  EventBuilder::clear();
}

int EbAppBase::resetCounters()
{
  _bufferCnt = 0;
  if (_fixupSrc)  _fixupSrc->clear();
  if (_ctrbSrc)   _ctrbSrc ->clear();

  return 0;
}

int EbAppBase::startConnection(const std::string& ifAddr,
                               std::string&       port,
                               unsigned           nLinks)
{
  int rc = linksStart(_transport, ifAddr, port, nLinks, "DRP");
  if (rc)  return rc;

  return 0;
}

int EbAppBase::connect(const EbParams& prms, size_t inpSizeGuess)
{
  unsigned nCtrbs = std::bitset<64>(prms.contributors).count();
  std::map<std::string, std::string> labels{{"instrument", prms.instrument},
                                            {"partition", std::to_string(prms.partition)},
                                            {"detname", prms.alias},
                                            {"alias", prms.alias},
                                            {"eb", _pfx}};

  _links        .resize(nCtrbs);
  _region       .resize(nCtrbs);
  _regSize      .resize(nCtrbs);
  _bufRegSize   .resize(nCtrbs);
  _maxTrSize    .resize(nCtrbs);
  _maxBufSize   .resize(nCtrbs);
  _id           = prms.id;
  _contributors = prms.contributors;
  _contract     = prms.contractors;
  _fixupSrc     = _exporter->histogram("EB_FxUpSc", labels, nCtrbs);
  _ctrbSrc      = _exporter->histogram("EB_CtrbSc", labels, nCtrbs); // Revisit: For testing

  int rc = linksConnect(_transport, _links, "DRP");
  if (rc)  return rc;

  // Set up a guess at the RDMA region now that we know the number of Contributors
  // If it's too small, it will be corrected during Configure
  for (unsigned i = 0; i < nCtrbs; ++i)
  {
    if (!_region[i])                    // No need to guess again
    {
      // Make a guess at the size of the Input region
      size_t regSizeGuess = (inpSizeGuess * _maxBuffers * _maxEntries +
                             roundUpSize(NUM_TRANSITION_BUFFERS * prms.maxTrSize[i]));
      //printf("*** EAB::connect: region %p, regSize %zu, regSizeGuess %zu\n",
      //       _region[i], _regSize[i], regSizeGuess);

      _region[i] = allocRegion(regSizeGuess);
      if (!_region[i])
      {
        logging::error("%s:\n  "
                       "No memory found for Input MR for %s[%d] of size %zd",
                       __PRETTY_FUNCTION__, "DRP", i, regSizeGuess);
        return ENOMEM;
      }

      // Save the allocated size, which may be more than the required size
      _regSize[i] = regSizeGuess;
    }

    //printf("*** EAB::connect: region %p, regSize %zu\n", _region[i], _regSize[i]);
    rc = _transport.setupMr(_region[i], _regSize[i]);
    if (rc)  return rc;
  }

  return 0;
}

int EbAppBase::configure(const EbParams& prms)
{
  int rc = _linksConfigure(prms, _links, _id, "DRP");
  if (rc)  return rc;

  for (unsigned buf = 0; buf < NUM_TRANSITION_BUFFERS; ++buf)
  {
    for (auto link : _links)
    {
      uint64_t imm  = ImmData::value(ImmData::Transition, _id, buf);

      if (unlikely(_verbose >= VL_EVENT))
        printf("EbAp posts transition buffer index %u to src %2u, %08lx\n",
               buf, link->id(), imm);

      rc = link->post(nullptr, 0, imm);
      if (rc)
      {
        logging::error("%s:\n  Failed to post buffer number to DRP ID %d: rc %d, imm %08lx",
                       __PRETTY_FUNCTION__, link->id(), rc, imm);
      }
    }
  }

  return 0;
}

int EbAppBase::_linksConfigure(const EbParams&            prms,
                               std::vector<EbLfSvrLink*>& links,
                               unsigned                   id,
                               const char*                peer)
{
  std::vector<EbLfSvrLink*> tmpLinks(links.size());

  for (auto link : links)
  {
    auto   t0(std::chrono::steady_clock::now());
    int    rc;
    size_t regSize;
    if ( (rc = link->prepare(id, &regSize, peer)) )
    {
      logging::error("%s:\n  Failed to prepare link with %s ID %d",
                     __PRETTY_FUNCTION__, peer, link->id());
      return rc;
    }
    unsigned rmtId     = link->id();
    tmpLinks[rmtId]    = link;

    _bufRegSize[rmtId] = regSize;
    _maxBufSize[rmtId] = regSize / (_maxBuffers * _maxEntries);
    _maxTrSize[rmtId]  = prms.maxTrSize[rmtId];
    regSize           += roundUpSize(NUM_TRANSITION_BUFFERS * _maxTrSize[rmtId]);  // Ctrbs don't have a transition space

    // Allocate the region, and reallocate if the required size is larger
    if (regSize > _regSize[rmtId])
    {
      if (_region[rmtId])  free(_region[rmtId]);

      _region[rmtId] = allocRegion(regSize);
      if (!_region[rmtId])
      {
        logging::error("%s:\n  "
                       "No memory found for Input MR for %s ID %d of size %zd",
                       __PRETTY_FUNCTION__, peer, rmtId, regSize);
        return ENOMEM;
      }

      // Save the allocated size, which may be more than the required size
      _regSize[rmtId] = regSize;
    }

    //printf("*** EAB::cfg: region %p, regSize %zu\n", _region[rmtId], regSize);
    if ( (rc = link->setupMr(_region[rmtId], regSize, peer)) )
    {
      logging::error("%s:\n  Failed to set up Input MR for %s ID %d, "
                     "%p:%p, size %zd", __PRETTY_FUNCTION__, peer, rmtId,
                     _region[rmtId], static_cast<char*>(_region[rmtId]) + regSize, regSize);
      return rc;
    }

    auto t1 = std::chrono::steady_clock::now();
    auto dT = std::chrono::duration_cast<ms_t>(t1 - t0).count();
    logging::info("Inbound link with %s ID %d configured in %lu ms",
                  peer, rmtId, dT);
  }

  links = tmpLinks;                     // Now in remote ID sorted order

  return 0;
}

int EbAppBase::process()
{
  int rc;

  // Pend for an input datagram and pass it to the event builder
  uint64_t  data;
  const int msTmo = 100;
  if ( (rc = _transport.pend(&data, msTmo)) < 0)
  {
    // Time out incomplete events
    if (rc == -FI_ETIMEDOUT)  EventBuilder::expired();
    else logging::error("%s:\n  pend() error %d\n", __PRETTY_FUNCTION__, rc);
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

  if (src != idg->xtc.src.value())
    logging::warning("Link src (%d) != dgram src (%d)", src, idg->xtc.src.value());

  _ctrbSrc->observe(double(src));       // Revisit: For testing

  if (unlikely(_verbose >= VL_BATCH))
  {
    unsigned    env = idg->env;
    uint64_t    pid = idg->pulseId();
    unsigned    ctl = idg->control();
    const char* svc = TransitionId::name(idg->service());
    printf("EbAp rcvd %9lu %15s[%8u]   @ "
           "%16p, ctl %02x, pid %014lx, env %08x,            src %2u, data %08lx, lnk %p, src %2u\n",
           _bufferCnt, svc, idx, idg, ctl, pid, env, lnk->id(), data, lnk, src);
  }

  // Tr space bufSize value is irrelevant since maxEntries will be 1 for that case
  EventBuilder::process(idg, _maxBufSize[src], data);

  return 0;
}

void EbAppBase::post(const EbDgram* const* begin, const EbDgram** const end)
{
  for (auto pdg = begin; pdg < end; ++pdg)
  {
    auto     idg = *pdg;
    unsigned src = idg->xtc.src.value();
    auto     lnk = _links[src];
    size_t   ofs = lnk->lclOfs(reinterpret_cast<const void*>(idg));
    unsigned buf = (ofs - _bufRegSize[src]) / _maxTrSize[src];
    uint64_t imm = ImmData::value(ImmData::Transition, _id, buf);

    if (unlikely(_verbose >= VL_EVENT))
      printf("EbAp posts transition buffer index %u to src %2u, %08lx\n",
             buf, src, imm);

    int rc = lnk->post(nullptr, 0, imm);
    if (rc)
    {
      logging::error("%s:\n  Failed to post buffer number to DRP ID %d: rc %d, imm %08lx",
                     __PRETTY_FUNCTION__, src, rc, imm);
    }
  }
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
  event->damage(Damage::DroppedContribution);

  if (!event->creator()->isEvent())
  {
    logging::warning("Fixup %s, %014lx, size %zu, source %d\n",
                     TransitionId::name(event->creator()->service()),
                     event->sequence(), event->size(), srcId);
  }

  _fixupSrc->observe(double(srcId));
}
