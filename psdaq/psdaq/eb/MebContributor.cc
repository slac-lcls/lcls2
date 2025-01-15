#include "MebContributor.hh"

#include "Endpoint.hh"
#include "EbLfClient.hh"

#include "utilities.hh"

#include "psalg/utils/SysLog.hh"
#include "psdaq/service/MetricExporter.hh"
#include "psdaq/service/EbDgram.hh"

#include <string.h>
#include <cstdint>
#include <string>

#define UNLIKELY(expr)  __builtin_expect(!!(expr), 0)
#define LIKELY(expr)    __builtin_expect(!!(expr), 1)

using namespace XtcData;
using namespace Pds::Eb;
using logging  = psalg::SysLog;
using ms_t     = std::chrono::milliseconds;


MebContributor::MebContributor(const MebCtrbParams& prms) :
  _prms       (prms),
  _maxEvSize  (roundUpSize(prms.maxEvSize)),
  _maxTrSize  (prms.maxTrSize),
  _transport  (prms.verbose, prms.kwargs),
  _id         (-1),
  _enabled    (false),
  _verbose    (prms.verbose),
  _previousPid(0),
  _eventCount (0),
  _trCount    (0)
{
}

int MebContributor::resetCounters()
{
  _eventCount = 0;
  _trCount    = 0;

  return 0;
}

void MebContributor::shutdown()
{
  if (!_links.empty())                  // Avoid shutting down if already done
  {
    unconfigure();
    disconnect();
  }
}

void MebContributor::disconnect()
{
  for (auto link : _links)  _transport.disconnect(link);
  _links.clear();

  _id = -1;
}

void MebContributor::unconfigure()
{
  _enabled = false;
}

int MebContributor::_setupMetrics(const std::shared_ptr<MetricExporter> exporter)
{
  std::map<std::string, std::string> labels{{"instrument", _prms.instrument},
                                            {"partition", std::to_string(_prms.partition)},
                                            {"detname", _prms.detName},
                                            {"detseg", std::to_string(_prms.detSegment)},
                                            {"alias", _prms.alias}};
  exporter->add("MCtbO_EvCt",  labels, MetricType::Counter, [&](){ return _eventCount;          });
  exporter->add("MCtbO_TrCt",  labels, MetricType::Counter, [&](){ return _trCount;             });
  exporter->add("MCtbO_TxPdg", labels, MetricType::Gauge,   [&](){ return _transport.posting(); });
  exporter->add("MCtbO_RxPdg", labels, MetricType::Gauge,   [&](){ return _transport.pending(); });

  return 0;
}

int MebContributor::connect(const std::shared_ptr<MetricExporter> exporter)
{
  if (exporter)
  {
    int rc = _setupMetrics(exporter);
    if (rc)  return rc;
  }

  _links      .resize(_prms.addrs.size());
  _region     .resize(_links.size());
  _regSize    .resize(_links.size());
  _bufRegSize .resize(_links.size());
  _trBuffers  .resize(_links.size());
  _id         = _prms.id;

  int rc = linksConnect(_transport, _links, _prms.addrs, _prms.ports, _id, "MEB");
  if (rc)  return rc;

  return 0;
}

int MebContributor::configure()
{
  int rc = _linksConfigure(_prms, _links, "MEB");
  if (rc)  return rc;

  // Code added here involving the links must be coordinated with the other side

  for (auto link : _links)
  {
    auto& lst = _trBuffers[link->id()];
    lst.clear();

    for (unsigned buf = 0; buf < MEB_TR_BUFFERS; ++buf)
    {
      lst.push_back(buf);
    }
  }

  _enabled = true;

  return 0;
}

int MebContributor::_linksConfigure(const MebCtrbParams&       prms,
                                    std::vector<EbLfCltLink*>& links,
                                    const char*                peer)
{
  // @todo: This could be done during Connect

  // Set up one region per MEB
  for (auto link : links)
  {
    auto     t0{std::chrono::steady_clock::now()};
    unsigned rmtId = link->id();

    size_t regSize = prms.maxEvents[rmtId] * _maxEvSize; // Needs MEB's connect_info
    _bufRegSize[rmtId] = regSize;
    regSize += MEB_TR_BUFFERS * _maxTrSize;

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

    int rc = link->prepare(_region[rmtId], _regSize[rmtId], _maxEvSize, "MEB");
    if (rc)
    {
      logging::error("%s:\n  Failed to prepare link with %s ID %d",
                     __PRETTY_FUNCTION__, peer, link->id());
      return rc;
    }

    auto t1 = std::chrono::steady_clock::now();
    auto dT = std::chrono::duration_cast<ms_t>(t1 - t0).count();
    auto rb = _region[rmtId];
    auto re = (char*)rb + _regSize[rmtId];
    logging::info("Outbound link with %3s ID %2d, %10p : %10p (%08zx), configured in %4lu ms",
                  peer, link->id(), rb, re, _regSize[rmtId], dT);
  }

  return 0;
}

int MebContributor::post(const EbDgram* ddg, uint32_t destination)
{
  // To avoid modifying the source data, we use the NoResponse bit below to get
  // the EOL bit set by EbAppBase
  //ddg->setEOL();                        // Set end-of-list marker

  uint64_t     pid    = ddg->pulseId();
  unsigned     dst    = ImmData::src(destination);
  uint32_t     idx    = ImmData::idx(destination);
  size_t       sz     = sizeof(*ddg) + ddg->xtc.sizeofPayload();
  unsigned     offset = idx * _maxEvSize;
  EbLfCltLink* link   = _links[dst];
  uint32_t     data   = ImmData::value(ImmData::NoResponse_Buffer, _id, idx);
  void*        buffer = (char*)(_region[dst]) + offset;
  memcpy(buffer, ddg, sz);  // Copy the datagram into the intermediate buffer

  if (UNLIKELY(sz > _maxEvSize))
  {
    logging::critical("L1Accept of size %zd is too big for target buffer of size %zd",
                      sz, _maxEvSize);
    abort();
  }

  if (UNLIKELY(ddg->xtc.src.value() != _id))
  {
    logging::critical("L1Accept src %u does not match DRP's ID %u: PID %014lx, sz, %zd, dest %08x, data %08x, ofs %08x",
                      ddg->xtc.src.value(), _id, pid, sz, destination, data, offset);
    abort();
  }

  bool print = false;
  if (UNLIKELY((buffer < _region[dst]) || ((char*)buffer + sz > (char*)_region[dst] + _bufRegSize[dst])))
  {
    logging::error("%s:\n  L1 dgram %p:%p falls outside of region limits %p:%p\n",
                   __PRETTY_FUNCTION__, buffer, (char*)buffer + sz, _region[dst], (char*)_region[dst] + _bufRegSize[dst]);
    print = true;
  }

  if (UNLIKELY(pid <= _previousPid))
  {
    logging::error("%s:\n  Pulse ID did not advance: %014lx <= %014lx, ts %u.%09u",
                   __PRETTY_FUNCTION__, pid, _previousPid, ddg->time.seconds(), ddg->time.nanoseconds());
    print = true;
  }

  if (UNLIKELY(print || (_verbose >= VL_BATCH)))
  {
    unsigned ctl    = ddg->control();
    uint32_t env    = ddg->env;
    void*    rmtAdx = (void*)link->rmtAdx(offset);
    printf("MebCtrb posts %9lu    monEvt [%8u]  @ "
           "%16p, ctl %02x, pid %014lx, env %08x, sz %6zd, MEB %2u @ %16p, data %08x\n",
           _eventCount, idx, ddg, ctl, pid, env, sz, link->id(), rmtAdx, data);
  }
  else
  {
    auto svc = ddg->service();
    if (svc != XtcData::TransitionId::L1Accept) {
      void* rmtAdx = (void*)link->rmtAdx(offset);
      if (svc != XtcData::TransitionId::SlowUpdate) {
        logging::info("MebCtrb   sent %s @ %u.%09u (%014lx) to MEB ID %u @ %16p (%08x + %u * %08zx)",
                      XtcData::TransitionId::name(svc),
                      ddg->time.seconds(), ddg->time.nanoseconds(),
                      ddg->pulseId(), dst, rmtAdx, 0, idx, _maxEvSize);
      }
      else {
        logging::debug("MebCtrb   sent %s @ %u.%09u (%014lx) to MEB ID %u @ %16p (%08x + %u * %08zx)",
                       XtcData::TransitionId::name(svc),
                       ddg->time.seconds(), ddg->time.nanoseconds(),
                       ddg->pulseId(), dst, rmtAdx, 0, idx, _maxEvSize);
      }
    }
  }

  int rc = link->post(buffer, sz, offset, data);
  if (rc < 0)
  {
    uint64_t pid    = ddg->pulseId();
    unsigned ctl    = ddg->control();
    uint32_t env    = ddg->env;
    void*    rmtAdx = (void*)link->rmtAdx(offset);
    logging::critical("%s:\n  Failed to post monEvt [%8u]  @ "
                      "%16p, ctl %02x, pid %014lx, env %08x, sz %6zd, MEB %2u @ %16p, data %08x, rc %d\n",
                      __PRETTY_FUNCTION__, idx, ddg, ctl, pid, env, sz, link->id(), rmtAdx, data, rc);
    abort();
  }

  _previousPid = pid;
  ++_eventCount;

  return 0;
}

// This is the same as in TebContributor as we have no good common place for it
static int _getTrBufIdx(EbLfLink* lnk, MebContributor::listU32_t& lst, uint32_t& idx)
{
  // Try to replenish the transition buffer index list
  while (true)
  {
    uint64_t imm;
    int rc = lnk->poll(&imm);           // Attempt to get a free buffer index
    if (rc)  break;
    if ((ImmData::flg(imm) != ImmData::NoResponse_Transition) ||
        (ImmData::src(imm) != lnk->id()))
      logging::error("%s: 1\n  "
                     "Flags %u != %u and/or source %u != %u in immediate data: %08lx\n",
                     __PRETTY_FUNCTION__, ImmData::flg(imm), ImmData::NoResponse_Transition,
                     ImmData::src(imm), lnk->id(), imm);
    lst.push_back(ImmData::idx(imm));
  }

  // If the list is still empty, wait for one
  if (lst.empty())
  {
    uint64_t imm;
    unsigned tmo = 5000;
    int rc = lnk->poll(&imm, tmo);      // Wait for a free buffer index
    if (rc)  return rc;
    idx = ImmData::idx(imm);
    if ((ImmData::flg(imm) != ImmData::NoResponse_Transition) ||
        (ImmData::src(imm) != lnk->id()))
      logging::error("%s: 1\n  "
                     "Flags %u != %u and/or source %u != %u in immediate data: %08lx\n",
                     __PRETTY_FUNCTION__, ImmData::flg(imm), ImmData::NoResponse_Transition,
                     ImmData::src(imm), lnk->id(), imm);
    return 0;
  }

  // Return the index at the head of the list
  idx = lst.front();
  lst.pop_front();

  return 0;
}

int MebContributor::post(const EbDgram* dgram)
{
  // To avoid modifying the source data, we use the NoResponse bit below to get
  // the EOL bit set by EbAppBase
  //dgram->setEOL();                        // Set end-of-list marker

  size_t sz  = sizeof(*dgram) + dgram->xtc.sizeofPayload();
  auto   pid = dgram->pulseId();
  auto   svc = dgram->service();
  bool   print = false;

  if (sz > _maxTrSize)
  {
    logging::critical("%s transition of size %zd is too big for target buffer of size %zd",
                      TransitionId::name(svc), sz, _maxTrSize);
    abort();
  }

  if (dgram->xtc.src.value() != _id)
  {
    logging::critical("%s transition src %u does not match DRP's ID %u for PID %014lx",
                      TransitionId::name(svc), dgram->xtc.src.value(), _id, pid);
    abort();
  }

  if (UNLIKELY(pid <= _previousPid))
  {
    logging::error("%s:\n  Pulse ID did not advance: %014lx <= %014lx, ts %u.%09u",
                   __PRETTY_FUNCTION__, pid, _previousPid, dgram->time.seconds(), dgram->time.nanoseconds());
    print = true;
  }
  _previousPid = pid;

  for (auto link : _links)
  {
    unsigned src = link->id();
    uint32_t idx;
    int rc = _getTrBufIdx(link, _trBuffers[src], idx);
    if (rc)
    {
      auto ts  = dgram->time;
      logging::critical("%s:\n  No transition buffer index received from MEB ID %u "
                        "needed for %s (%014lx, %9u.%09u): rc %d",
                        __PRETTY_FUNCTION__, src, TransitionId::name(svc), pid, ts.seconds(), ts.nanoseconds(), rc);
      abort();
    }

    uint64_t offset = _bufRegSize[src] + idx * _maxTrSize;
    uint32_t data   = ImmData::value(ImmData::NoResponse_Transition, _id, idx);
    void*    buffer = (char*)(_region[src]) + offset;
    memcpy(buffer, dgram, sz); // Copy the datagram into the intermediate buffer

    if (UNLIKELY((buffer < (char*)_region[src] + _bufRegSize[src]) || ((char*)buffer + sz > (char*)_region[src] + _regSize[src])))
    {
      logging::error("%s:\n  Tr dgram %p:%p falls outside of region limits %p:%p\n",
                     __PRETTY_FUNCTION__, buffer, (char*)buffer + sz, (char*)_region[src] + _bufRegSize[src], (char*)_region[src] + _regSize[src]);
      print = true;
    }

    if (UNLIKELY(print || (_verbose >= VL_BATCH)))
    {
      printf("MebCtrb rcvd transition buffer           [%2u] @ "
             "%16p, ofs %016lx = %08zx + %2u * %08zx,     src %2u\n",
             idx, (void*)link->rmtAdx(0), offset, _bufRegSize[src], idx, _maxTrSize, src);

      unsigned ctl    = dgram->control();
      uint32_t env    = dgram->env;
      void*    rmtAdx = (void*)link->rmtAdx(offset);
      printf("MebCtrb posts %9lu %15s       @ "
             "%16p, ctl %02x, pid %014lx, env %08x, sz %6zd, MEB %2u @ %16p, data %08x\n",
             _trCount, TransitionId::name(svc), dgram, ctl, pid, env, sz, src, rmtAdx, data);
      print = false;                    // Print only once per call
    }
    else
    {
      auto svc = dgram->service();
      if (svc != XtcData::TransitionId::L1Accept) {
        void* rmtAdx = (void*)link->rmtAdx(offset);
        if (svc != XtcData::TransitionId::SlowUpdate) {
          logging::info("MebCtrb   sent %s @ %u.%09u (%014lx) to MEB ID %u @ %16p (%08zx + %u * %08zx)",
                        XtcData::TransitionId::name(svc),
                        dgram->time.seconds(), dgram->time.nanoseconds(),
                        dgram->pulseId(), src, rmtAdx, _bufRegSize[src], idx, _maxTrSize);
        }
        else {
          logging::debug("MebCtrb   sent %s @ %u.%09u (%014lx) to MEB ID %u @ %16p (%08zx + %u * %08zx)",
                         XtcData::TransitionId::name(svc),
                         dgram->time.seconds(), dgram->time.nanoseconds(),
                         dgram->pulseId(), src, rmtAdx, _bufRegSize[src], idx, _maxTrSize);
        }
      }
    }

    rc = link->post(buffer, sz, offset, data); // Not a batch; Continue on error
    if (rc)
    {
      logging::error("%s:\n  Failed to post buffer number to MEB ID %u: rc %d, data %08x",
                     __PRETTY_FUNCTION__, src, rc, data);
    }
  }

  ++_trCount;

  return 0;
}
