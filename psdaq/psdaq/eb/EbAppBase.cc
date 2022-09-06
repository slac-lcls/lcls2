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

#define UNLIKELY(expr)  __builtin_expect(!!(expr), 0)
#define LIKELY(expr)    __builtin_expect(!!(expr), 1)

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
                     const unsigned          msTimeout) :
  EventBuilder (msTimeout, prms.verbose),
  _transport   (prms.verbose, prms.kwargs),
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
  exporter->constant("EB_EvPlDp", labels, eventPoolDepth());

  exporter->add("EB_EvAlCt", labels, MetricType::Counter, [&](){ return  eventAllocCnt();     });
  exporter->add("EB_EvFrCt", labels, MetricType::Counter, [&](){ return  eventFreeCnt();      });
  exporter->add("EB_EvOcCt", labels, MetricType::Gauge,   [&](){ return  eventOccCnt();       });
  exporter->add("EB_EpOcCt", labels, MetricType::Gauge,   [&](){ return  epochOccCnt();       });
  exporter->add("EB_RxPdg",  labels, MetricType::Gauge,   [&](){ return _transport.pending(); });
  exporter->add("EB_TxPdg",  labels, MetricType::Gauge,   [&](){ return _transport.posting(); });
  exporter->add("EB_BfInCt", labels, MetricType::Counter, [&](){ return _bufferCnt;           }); // Inbound
  exporter->add("EB_ToEvCt", labels, MetricType::Counter, [&](){ return  timeoutCnt();        });
  exporter->add("EB_FxUpCt", labels, MetricType::Counter, [&](){ return  fixupCnt();          });
  exporter->add("EB_CbMsMk", labels, MetricType::Gauge,   [&](){ return  missing();           });
  exporter->add("EB_EvAge",  labels, MetricType::Gauge,   [&](){ return  eventAge();          });
  exporter->add("EB_dTime",  labels, MetricType::Gauge,   [&](){ return  ebTime();            });
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

int EbAppBase::resetCounters()
{
  _bufferCnt = 0;
  if (_fixupSrc)  _fixupSrc->clear();
  if (_ctrbSrc)   _ctrbSrc ->clear();
  EventBuilder::resetCounters();

  return 0;
}

void EbAppBase::shutdown()
{
  _transport.shutdown();
}

void EbAppBase::disconnect()
{
  for (auto link : _links)  _transport.disconnect(link);
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
  if (!_links.empty())                  // Avoid dumping again if already done
    EventBuilder::dump(0);
  EventBuilder::clear();
}

int EbAppBase::startConnection(const std::string& ifAddr,
                               std::string&       port,
                               unsigned           nLinks)
{
  int rc = _transport.listen(ifAddr, port, nLinks);
  if (rc)
  {
    logging::error("%s:\n  Failed to initialize %s EbLfServer on %s:%s",
                   __PRETTY_FUNCTION__, "DRP", ifAddr.c_str(), port.c_str());
    return rc;
  }

  return 0;
}

int EbAppBase::connect(const EbParams& prms, size_t inpSizeGuess)
{
  int      rc;
  unsigned nCtrbs = std::bitset<64>(prms.contributors).count();

  // Initialize the event builder
  auto duration = prms.maxEntries;
  _maxEntries   = prms.maxEntries;
  _maxEvBuffers = prms.maxBuffers / prms.maxEntries;
  _maxTrBuffers = TEB_TR_BUFFERS;
  rc = initialize(_maxEvBuffers + _maxTrBuffers, _maxEntries, nCtrbs, duration);
  if (rc)  return rc;

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
  _idxSrcs      = prms.indexSources;
  _contract     = prms.contractors;
  _fixupSrc     = _exporter->histogram("EB_FxUpSc", labels, nCtrbs);
  _ctrbSrc      = _exporter->histogram("EB_CtrbSc", labels, nCtrbs); // Revisit: For testing

  for (auto i = 0u; i < nCtrbs; ++i)
  {
    // Pass loop index by value or it will be out of scope when lambda runs
    _exporter->add("EB_arrTime" + std::to_string(i), labels, MetricType::Gauge, [=](){ return  arrTime(i);});
  }

  rc = linksConnect(_transport, _links, _id, "DRP");
  if (rc)  return rc;

  // Assume an existing region is already appropriately sized, else make a guess
  // at a suitable RDMA region to avoid spending time in Configure.
  // If it's too small, it will be corrected during Configure
  if (inpSizeGuess)                     // Disable by providing 0
  {
    for (auto link : _links)
    {
      unsigned rmtId  = link->id();
      if (!_region[rmtId])                  // No need to guess again
      {
        // Make a guess at the size of the Input region
        size_t regSizeGuess = (inpSizeGuess * prms.numBuffers[rmtId] +
                               _maxTrBuffers * prms.maxTrSize[rmtId]);

        _region[rmtId] = allocRegion(regSizeGuess);
        if (!_region[rmtId])
        {
          logging::error("%s:\n  "
                         "No memory found for Input MR for %s ID %u of size %zd",
                         __PRETTY_FUNCTION__, "DRP", rmtId, regSizeGuess);
          return ENOMEM;
        }

        // Save the allocated size, which may be more than the required size
        _regSize[rmtId] = regSizeGuess;
      }

      rc = _transport.setupMr(_region[rmtId], _regSize[rmtId]);
      if (rc)  return rc;
    }
  }

  return 0;
}

int EbAppBase::configure(const EbParams& prms)
{
  int rc = _linksConfigure(prms, _links, "DRP");
  if (rc)  return rc;

  // Code added here involving the links must be coordinated with the other side

  return 0;
}

int EbAppBase::_linksConfigure(const EbParams&            prms,
                               std::vector<EbLfSvrLink*>& links,
                               const char*                peer)
{
  for (auto link : links)
  {
    auto   t0{std::chrono::steady_clock::now()};
    int    rc;
    size_t regEntrySize;
    if ( (rc = link->prepare(&regEntrySize, peer)) )
    {
      logging::error("%s:\n  Failed to prepare link with %s ID %d",
                     __PRETTY_FUNCTION__, peer, link->id());
      return rc;
    }
    unsigned rmtId     = link->id();
    size_t regSize     = regEntrySize * prms.numBuffers[rmtId];
    _bufRegSize[rmtId] = regSize;
    _maxBufSize[rmtId] = regEntrySize;
    _maxTrSize[rmtId]  = prms.maxTrSize[rmtId];
    regSize           += _maxTrBuffers * _maxTrSize[rmtId];  // Ctrbs don't have a transition space

    // Reallocate the region if the required size has changed
    if (regSize != _regSize[rmtId])
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

      _regSize[rmtId] = regSize;
    }

    if ( (rc = link->setupMr(_region[rmtId], regSize, peer)) )
    {
      logging::error("%s:\n  Failed to set up Input MR for %s ID %d, "
                     "%p:%p, size %zd", __PRETTY_FUNCTION__, peer, rmtId,
                     _region[rmtId], static_cast<char*>(_region[rmtId]) + regSize, regSize);
      return rc;
    }

    auto t1{std::chrono::steady_clock::now()};
    auto dT{std::chrono::duration_cast<ms_t>(t1 - t0).count()};
    auto rs = _regSize[rmtId];                // Size of the whole MR
    auto rb = _region[rmtId];                 // Batch/buffer space
    auto rt = (char*)rb + _bufRegSize[rmtId]; // Transition space
    auto re = (char*)rb + rs;                 // End
    logging::info("Inbound  link with %3s ID %2d, %10p : %10p : %10p (%08zx), configured in %4lu ms",
                  peer, rmtId, rb, rt, re, rs, dT);
  }

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
    if (rc == -FI_ETIMEDOUT)
    {
      // This is called when contributions have ceased flowing
      EventBuilder::expired();          // Time out incomplete events
    }
    else if (_transport.pollEQ() == -FI_ENOTCONN)
      rc = -FI_ENOTCONN;
    else
      logging::error("%s:\n  pend() error %d (%s)",
                     __PRETTY_FUNCTION__, rc, strerror(-rc));
    return rc;
  }

  unsigned       flg = ImmData::flg(data);
  unsigned       src = ImmData::src(data);
  unsigned       idx = ImmData::idx(data);
  EbLfSvrLink*   lnk = _links[src];
  size_t         ofs = (ImmData::buf(flg) == ImmData::Buffer)
                     ? (                   idx * _maxBufSize[src]) // In batch/buffer region
                     : (_bufRegSize[src] + idx * _maxTrSize[src]); // Tr region for non-selected EB is after batch/buffer region
  const EbDgram* idg = static_cast<EbDgram*>(lnk->lclAdx(ofs));    // Or, (char*)(_region[src]) + ofs;

  // "Non-selected" TEBs receive only single dgrams that are transitions needing
  // to have their EOL flag set to avoid the EB iterating to the next buffer.
  // This isn't done on the DRPs since these dgrams are also sent to the
  // "selected" TEB, which must not be caused to stop iterating over its batch
  // prematurely.  MEBs receive only single dgrams that are the source data,
  // which shouldn't be modified, so we userp the NoResponse bit (which isn't
  // used by MEBs) to indicate it should be done here.
  if (flg & ImmData::NoResponse)  idg->setEOL();

  if (src != idg->xtc.src.value())
  {
    logging::error("Link src (%d) != dgram src (%d)", src, idg->xtc.src.value());
    _verbose = VL_EVENT;
  }

  _ctrbSrc->observe(double(src));       // Revisit: For testing

  if (UNLIKELY(_verbose >= VL_BATCH))
  {
    unsigned    env = idg->env;
    uint64_t    pid = idg->pulseId();
    unsigned    ctl = idg->control();
    const char* svc = TransitionId::name(idg->service());
    printf("EbAp rcvd %9lu %15s[%8u]   @ "
           "%16p, ctl %02x, pid %014lx, env %08x,            src %2u, data %08lx, lnk[%2u] %p, ID %2u\n",
           _bufferCnt, svc, idx, idg, ctl, pid, env, idg->xtc.src.value(), data, src, lnk, lnk->id());
  }

  auto svc = idg->service();
  if (svc != XtcData::TransitionId::L1Accept) {
    auto base = (ImmData::buf(flg) == ImmData::Buffer) ?           0      : _bufRegSize[src];
    auto size = (ImmData::buf(flg) == ImmData::Buffer) ? _maxBufSize[src] : _maxTrSize[src];
    if (svc != XtcData::TransitionId::SlowUpdate) {
      logging::info("EbAppBase  saw %15s @ %u.%09u (%014lx) from DRP ID %2u @ %16p (%08zx + %2u * %08zx)",
                    XtcData::TransitionId::name(svc),
                    idg->time.seconds(), idg->time.nanoseconds(),
                    idg->pulseId(), src, idg, base, idx, size);
    }
    else {
      logging::debug("EbAppBase  saw %15s @ %u.%09u (%014lx) from DRP ID %2u @ %16p (%08zx + %2u * %08zx)",
                     XtcData::TransitionId::name(svc),
                     idg->time.seconds(), idg->time.nanoseconds(),
                     idg->pulseId(), src, idg, base, idx, size);
    }
  }

  // Tr space bufSize value is irrelevant since idg has EOL set in that case
  if ((_idxSrcs & (1ull << src)) == 0)  data = 0;
  EventBuilder::process(idg, _maxBufSize[src], data);

  ++_bufferCnt;

  return 0;
}

void EbAppBase::post(const EbDgram* const* begin, const EbDgram** const end)
{
  for (auto pdg = begin; pdg < end; ++pdg)
  {
    auto     idg = *pdg;
    unsigned src = idg->xtc.src.value();
    auto     lnk = _links[src];
    size_t   ofs = lnk->lclOfs(idg);
    unsigned idx = (ofs - _bufRegSize[src]) / _maxTrSize[src];
    uint64_t imm = ImmData::value(ImmData::Transition, _id, idx);

    if (UNLIKELY(_verbose >= VL_EVENT))
      printf("EbAp posts transition buffer index %u to src %2u, %08lx\n",
             idx, src, imm);

    int rc = lnk->post(imm);
    if (rc)
    {
      logging::error("%s:\n  Failed to post transition buffer index %u to DRP ID %u: rc %d, imm %08lx",
                     __PRETTY_FUNCTION__, idx, src, rc, imm);
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
    logging::warning("Fixup %s, %014lx, size %zu, source %d",
                     TransitionId::name(event->creator()->service()),
                     event->sequence(), event->size(), srcId);
  }

  _fixupSrc->observe(double(srcId));
}
