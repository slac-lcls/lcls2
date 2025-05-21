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

static unsigned _ebTimeout(const EbParams& prms)
{
  if (prms.kwargs.find("eb_timeout") != prms.kwargs.end())
    return std::stoul(const_cast<EbParams&>(prms).kwargs["eb_timeout"]);

  const_cast<EbParams&>(prms).kwargs["eb_timeout"] = std::to_string(EB_TMO_MS);
  return EB_TMO_MS;
}

EbAppBase::EbAppBase(const EbParams&    prms,
                     const std::string& pfx) :
  EventBuilder(_ebTimeout(prms), prms.verbose),
  _transport  (prms.verbose, prms.kwargs),
  _verbose    (prms.verbose),
  _lastPid    (0),
  _bufferCnt  (0),
  _id         (-1),
  _pfx        (pfx),
  _prms       (prms)
{
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
  // If connect() ran but the system didn't get into the Connected state,
  // there won't be a Disconnect transition, so disconnect() here
  disconnect();                         // Does no harm if already done

  _transport.shutdown();
}

void EbAppBase::disconnect()
{
  // If configure() ran but the system didn't get into the Configured state,
  // there won't be an Unconfigure transition, so unconfigure() here
  unconfigure();                        // Does no harm if already done

  for (auto link : _links)  _transport.disconnect(link);
  _links.clear();

  _id           = -1;
  _contract     .fill(0);
  _bufRegSize   .clear();
  _maxBufSize   .clear();
  _maxTrSize    .clear();
}

void EbAppBase::unconfigure()
{
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

int EbAppBase::_setupMetrics(const MetricExporter_t exporter)
{
  std::map<std::string, std::string> labels{{"instrument", _prms.instrument},
                                            {"partition", std::to_string(_prms.partition)},
                                            {"detname", _prms.alias},
                                            {"alias", _prms.alias},
                                            {"eb", _pfx}};
  exporter->add("EB_RxPdg",  labels, MetricType::Gauge,   [&](){ return _transport.pending(); });
  exporter->add("EB_TxPdg",  labels, MetricType::Gauge,   [&](){ return _transport.posting(); });
  exporter->add("EB_BfInCt", labels, MetricType::Counter, [&](){ return _bufferCnt;           }); // Inbound
  exporter->add("EB_ToEvCt", labels, MetricType::Counter, [&](){ return  timeoutCnt();        });
  exporter->add("EB_FxUpCt", labels, MetricType::Counter, [&](){ return  fixupCnt();          });
  exporter->add("EB_CbMsMk", labels, MetricType::Gauge,   [&](){ return  missing();           });
  exporter->add("EB_EvAge",  labels, MetricType::Gauge,   [&](){ return  eventAge();          });
  exporter->add("EB_dTime",  labels, MetricType::Gauge,   [&](){ return  ebTime();            });

  exporter->constant("EB_EvPlDp", labels, eventPoolDepth());

  exporter->add("EB_EpAlCt", labels, MetricType::Counter, [&](){ return epochAllocCnt(); });
  exporter->add("EB_EpFrCt", labels, MetricType::Counter, [&](){ return epochFreeCnt();  });
  exporter->add("EB_EvAlCt", labels, MetricType::Counter, [&](){ return eventAllocCnt(); });
  exporter->add("EB_EvFrCt", labels, MetricType::Counter, [&](){ return eventFreeCnt();  });
  exporter->add("EB_EvOcCt", labels, MetricType::Gauge,   [&](){ return eventOccCnt();   });
  exporter->add("EB_EpOcCt", labels, MetricType::Gauge,   [&](){ return epochOccCnt();   });

  unsigned nCtrbs = std::bitset<64>(_prms.contributors).count();
  for (auto i = 0u; i < nCtrbs; ++i)
  {
    // Pass loop index by value or it will be out of scope when lambda runs
    labels["ctrb"] = _prms.drps[i];
    exporter->add("EB_arrTime" + std::to_string(i), labels, MetricType::Gauge, [=,this](){ return arrTime(i); });
  }

  _fixupSrc = exporter->histogram("EB_FxUpSc", labels, nCtrbs);
  _ctrbSrc  = exporter->histogram("EB_CtrbSc", labels, nCtrbs); // Revisit: For testing

  return 0;
}

int EbAppBase::connect(unsigned maxTrBuffers, const MetricExporter_t exporter)
{
  int      rc;
  unsigned nCtrbs = std::bitset<64>(_prms.contributors).count();
  _links        .resize(nCtrbs);
  _region       .resize(nCtrbs);
  _regSize      .resize(nCtrbs);
  _bufRegSize   .resize(nCtrbs);
  _maxTrSize    .resize(nCtrbs);
  _maxBufSize   .resize(nCtrbs);
  _lastPid      .resize(nCtrbs);
  _id           = _prms.id;
  _idxSrcs      = _prms.indexSources;
  _contract     = _prms.contractors;

  // Initialize the event builder
  auto duration = _prms.maxEntries;
  _maxEntries   = _prms.maxEntries;
  _maxEvBuffers = (EB_TMO_MS / 1000) * (_prms.maxBuffers / _prms.maxEntries);
  _maxTrBuffers = maxTrBuffers;
  rc = initialize(_maxEvBuffers + _maxTrBuffers, _maxEntries, nCtrbs, duration);
  if (rc)  return rc;

  if (exporter)
  {
    rc = _setupMetrics(exporter);
    if (rc)  return rc;
  }

  rc = linksConnect(_transport, _links, _id, "DRP");
  if (rc)  return rc;

  return 0;
}

int EbAppBase::configure()
{
  int rc = _linksConfigure(_prms, _links, "DRP");
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
    size_t regEntrySize; // Trigger data size on TEB, max_ev_size[drpId] on MEB
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
    if (rc == -FI_EAGAIN)
    {
      // This is called when contributions have ceased flowing
      EventBuilder::expired();          // Time out incomplete events
    }
    else if (rc != -FI_ENOTCONN)
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
  auto           sz  = sizeof(*idg) + idg->xtc.sizeofPayload();
  void*          end;
  if (ImmData::buf(flg) == ImmData::Buffer)
  {
    // idg is first dgram in batch; set end to end of region if idg is within 1 batch size of it
    end = idx < _prms.numBuffers[src] - _maxEntries ? (char*)idg + _maxEntries * _maxBufSize[src]
                                                    : (char*)_region[src] + _bufRegSize[src];
  } else {
    end = (char*)idg + _maxTrSize[src];
  }

  // "Non-selected" TEBs receive only single dgrams that are transitions needing
  // to have their EOL flag set to avoid the EB iterating to the next buffer.
  // This isn't done on the DRPs since these dgrams are also sent to the
  // "selected" TEB, which must not be caused to stop iterating over its batch
  // prematurely.  MEBs receive only single dgrams that are the source data,
  // which shouldn't be modified, so we userp the NoResponse bit (which isn't
  // used by MEBs) to indicate it should be done here.
  if (ImmData::rsp(flg) == ImmData::NoResponse)  idg->setEOL();

  auto print = false;
  if (src != idg->xtc.src.value())
  {
    logging::error("%s:\n  Link src (%d) != dgram src (%d)", __PRETTY_FUNCTION__, src, idg->xtc.src.value());
    print = true;
  }
  if (ImmData::buf(flg) == ImmData::Buffer)
  {
    if (idx > _prms.numBuffers[src])
    {
      logging::error("%s:\n  Buffer index for src %d is out of range 0:%u: %u\n",
                     __PRETTY_FUNCTION__, src, _prms.numBuffers[src], idx);
      print = true;
    }
    if ((idg < _region[src]) || (end > ((char*)_region[src] + _bufRegSize[src])))
    {
      logging::error("%s:\n  Buffer %p:%p falls outside of region limits %p:%p\n",
                     __PRETTY_FUNCTION__, idg, end, _region[src], (char*)_region[src] + _bufRegSize[src]);
      print = true;
    }
    if (sz > _maxBufSize[src])
    {
      logging::error("%s:\n  Buffer's dgram %p, size %u overruns buffer of size %zu\n",
                     __PRETTY_FUNCTION__, idg, sz, _maxBufSize[src]);
      print = true;
    }
  }
  else
  {
    if (idx > _maxTrBuffers)
    {
      logging::error("%s:\n  Tr buffer index for src %d is out of range 0:%u: %u\n",
                     __PRETTY_FUNCTION__, src, _maxTrBuffers, idx);
      print = true;
    }
    if ((idg < (void*)((char*)_region[src] + _bufRegSize[src])) || (end > ((char*)_region[src] + _regSize[src])))
    {
      logging::error("%s:\n  Tr dgram %p:%p falls outside of region limits %p:%p\n",
                     __PRETTY_FUNCTION__, idg, end,
                     (char*)_region[src] + _bufRegSize[src], (char*)_region[src] + _regSize[src]);
      print = true;
    }
    if (sz > _maxTrSize[src])
    {
      logging::error("%s:\n  Tr dgram %p, size %u overruns buffer of size %zu\n",
                     __PRETTY_FUNCTION__, idg, sz, _maxTrSize[src]);
      print = true;
    }
  }
  if (idg->pulseId() <= _lastPid[src])
  {
    logging::error("%s:\n  Pulse ID for src %u did not advance: %014lx <= %014lx, ts %u.%09u",
                   __PRETTY_FUNCTION__, src, idg->pulseId(), _lastPid[src], idg->time.seconds(), idg->time.nanoseconds());
    print = true;
  }
  _lastPid[src] = idg->pulseId();

  _ctrbSrc->observe(double(src));       // Revisit: For testing

  if (UNLIKELY(print || (_verbose >= VL_BATCH)))
  {
    unsigned    env = idg->env;
    uint64_t    pid = idg->pulseId();
    unsigned    ctl = idg->control();
    const char* svc = TransitionId::name(idg->service());
    fprintf(stderr, "EbAp rcvd %9lu %15s[%8u]   @ "
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
  EventBuilder::process(idg, _maxBufSize[src], data, end);

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
    uint64_t imm = ImmData::value(ImmData::NoResponse_Transition, _id, idx);

    if (UNLIKELY(_verbose >= VL_EVENT))
      fprintf(stderr, "EbAp posts transition buffer index %u to src %2u, %08lx\n",
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

  if (fixupCnt() + timeoutCnt() < 100)
  {
    logging::warning("Fixup %s, %014lx, size %zu, source %d (%s)",
                     TransitionId::name(event->creator()->service()),
                     event->sequence(), event->size(),
                     srcId, _prms.drps[srcId].c_str());
  }

  _fixupSrc->observe(double(srcId));
}
