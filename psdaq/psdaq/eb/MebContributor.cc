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

using namespace XtcData;
using namespace Pds::Eb;
using logging  = psalg::SysLog;


MebContributor::MebContributor(const MebCtrbParams&            prms,
                               std::shared_ptr<MetricExporter> exporter) :
  _prms      (prms),
  _maxEvSize (roundUpSize(prms.maxEvSize)),
  _maxTrSize (prms.maxTrSize),
  _bufRegSize(prms.maxEvents * _maxEvSize),
  _transport (prms.verbose, prms.kwargs),
  _id        (-1),
  _enabled   (false),
  _verbose   (prms.verbose),
  _eventCount(0)
{
  std::map<std::string, std::string> labels{{"instrument", prms.instrument},
                                            {"partition", std::to_string(prms.partition)},
                                            {"alias", prms.alias}};
  exporter->add("MCtbO_EvCt",  labels, MetricType::Counter, [&](){ return _eventCount;          });
  exporter->add("MCtbO_TxPdg", labels, MetricType::Counter, [&](){ return _transport.pending(); });
}

int MebContributor::resetCounters()
{
  _eventCount = 0;

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

int MebContributor::connect(const MebCtrbParams& prms,
                            void*                region,
                            size_t               regSize)
{
  _links    .resize(prms.addrs.size());
  _trBuffers.resize(_links.size());
  _id       = prms.id;

  int rc = linksConnect(_transport, _links, prms.addrs, prms.ports, "MEB");
  if (rc)  return rc;

  //printf("*** MC::connect: region %p, regSize %zu\n", region, regSize);
  for (auto link : _links)
  {
    rc = link->setupMr(region, regSize);
    if (rc)  return rc;
  }

  return 0;
}

int MebContributor::configure(void*  region,
                              size_t regSize)
{
  //printf("*** MC::cfg: region %p, regSize %zu\n", region, regSize);
  int rc = linksConfigure(_links, _id, region, regSize, _bufRegSize, "MEB");
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

int MebContributor::post(const EbDgram* ddg, uint32_t destination)
{
  ddg->setEOL();                        // Set end-of-list marker

  unsigned     dst    = ImmData::src(destination);
  uint32_t     idx    = ImmData::idx(destination);
  size_t       sz     = sizeof(*ddg) + ddg->xtc.sizeofPayload();
  unsigned     offset = idx * _maxEvSize;
  EbLfCltLink* link   = _links[dst];
  uint32_t     data   = ImmData::value(ImmData::Buffer, _id, idx);

  if (sz > _maxEvSize)
  {
    logging::critical("L1Accept of size %zd is too big for target buffer of size %zd",
                      sz, _maxEvSize);
    abort();
  }

  if (ddg->xtc.src.value() != _id)
  {
    logging::critical("L1Accept src %u does not match DRP's ID %u: PID %014lx, sz, %zd, dest %08x, data %08x, ofs %08x",
                      ddg->xtc.src.value(), _id, ddg->pulseId(), sz, destination, data, offset);
    abort();
  }

  if (_verbose >= VL_BATCH)
  {
    uint64_t pid    = ddg->pulseId();
    unsigned ctl    = ddg->control();
    uint32_t env    = ddg->env;
    void*    rmtAdx = (void*)link->rmtAdx(offset);
    printf("MebCtrb posts %9lu    monEvt [%8u]  @ "
           "%16p, ctl %02x, pid %014lx, env %08x, sz %6zd, MEB %2u @ %16p, data %08x\n",
           _eventCount, idx, ddg, ctl, pid, env, sz, link->id(), rmtAdx, data);
  }

  if (int rc = link->post(ddg, sz, offset, data) < 0)  return rc;

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
    return 0;
  }

  // Return the index at the head of the list
  idx = lst.front();
  lst.pop_front();

  return 0;
}

int MebContributor::post(const EbDgram* dgram)
{
  dgram->setEOL();                        // Set end-of-list marker

  size_t sz  = sizeof(*dgram) + dgram->xtc.sizeofPayload();
  auto   svc = dgram->service();

  if (sz > _maxTrSize)
  {
    logging::critical("%s transition of size %zd is too big for target buffer of size %zd",
                      TransitionId::name(svc), sz, _maxTrSize);
    abort();
  }

  if (dgram->xtc.src.value() != _id)
  {
    logging::critical("%s transition src %u does not match DRP's ID %u for PID %014lx",
                      TransitionId::name(svc), dgram->xtc.src.value(), _id, dgram->pulseId());
    abort();
  }

  for (auto link : _links)
  {
    unsigned src = link->id();
    uint32_t idx;
    int rc = _getTrBufIdx(link, _trBuffers[src], idx);
    if (rc)
    {
      auto pid = dgram->pulseId();
      auto ts  = dgram->time;
      logging::critical("%s:\n  No transition buffer index received from MEB ID %u "
                        "needed for %s (%014lx, %9u.%09u): rc %d",
                        __PRETTY_FUNCTION__, src, TransitionId::name(svc), pid, ts.seconds(), ts.nanoseconds(), rc);
      abort();
    }

    uint64_t offset = _bufRegSize + idx * _maxTrSize;
    uint32_t data   = ImmData::value(ImmData::Transition, _id, idx);

    if (unlikely(_verbose >= VL_BATCH))
    {
      printf("MebCtrb rcvd transition buffer           [%2u] @ "
             "%16p, ofs %016lx = %08lx + %2u * %08lx,     src %2u\n",
             idx, (void*)link->rmtAdx(0), offset, _bufRegSize, idx, _maxTrSize, src);

      uint64_t pid    = dgram->pulseId();
      unsigned ctl    = dgram->control();
      uint32_t env    = dgram->env;
      void*    rmtAdx = (void*)link->rmtAdx(offset);
      printf("MebCtrb posts %9lu %15s       @ "
             "%16p, ctl %02x, pid %014lx, env %08x, sz %6zd, MEB %2u @ %16p, data %08x\n",
             _eventCount, TransitionId::name(svc), dgram, ctl, pid, env, sz, src, rmtAdx, data);
    }

    rc = link->post(dgram, sz, offset, data); // Not a batch; Continue on error
    if (rc)
    {
      logging::error("%s:\n  Failed to post buffer number to MEB ID %u: rc %d, data %08x",
                     __PRETTY_FUNCTION__, src, rc, data);
    }
  }

  ++_eventCount;                        // Revisit: Count these?

  return 0;
}
