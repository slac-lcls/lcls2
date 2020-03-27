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
  _maxEvSize (roundUpSize(prms.maxEvSize)),
  _maxTrSize (prms.maxTrSize),
  _bufRegSize(prms.maxEvents * _maxEvSize),
  _transport (prms.verbose),
  _links     (),
  _id        (-1),
  _verbose   (prms.verbose),
  _eventCount(0)
{
  std::map<std::string, std::string> labels{{"partition", std::to_string(prms.partition)}};
  exporter->add("MCtbO_EvCt",  labels, MetricType::Counter, [&](){ return _eventCount;          });
  exporter->add("MCtbO_TxPdg", labels, MetricType::Counter, [&](){ return _transport.pending(); });
}

int MebContributor::configure(const MebCtrbParams& prms,
                              void*                region,
                              size_t               size)
{
  _id         = prms.id;
  _eventCount = 0;
  _links.resize(prms.addrs.size());

  for (unsigned i = 0; i < prms.addrs.size(); ++i)
  {
    int            rc;
    const char*    addr = prms.addrs[i].c_str();
    const char*    port = prms.ports[i].c_str();
    EbLfCltLink*   link;
    const unsigned tmo(120000);         // Milliseconds
    if ( (rc = _transport.connect(&link, addr, port, _id, tmo)) )
    {
      logging::error("%s:\n  Error connecting to MEB at %s:%s\n",
                     __PRETTY_FUNCTION__, addr, port);
      return rc;
    }
    if ( (rc = link->prepare(region, size, _bufRegSize)) )
    {
      logging::error("%s:\n  Failed to prepare link with MEB at %s:%s\n",
                     __PRETTY_FUNCTION__, addr, port);
      return rc;
    }
    _links[link->id()] = link;

    logging::info("Outbound link with MEB ID %d connected\n", link->id());
  }

  return 0;
}

void MebContributor::shutdown()
{
  for (auto it = _links.begin(); it != _links.end(); ++it)
  {
    _transport.disconnect(*it);
  }
  _links.clear();

  _id = -1;
}

int MebContributor::post(const EbDgram* ddg, uint32_t destination)
{
  unsigned     dst    = ImmData::src(destination);
  uint32_t     idx    = ImmData::idx(destination);
  size_t       sz     = sizeof(*ddg) + ddg->xtc.sizeofPayload();
  unsigned     offset = idx * _maxEvSize;
  EbLfCltLink* link   = _links[dst];
  uint32_t     data   = ImmData::value(ImmData::Buffer, _id, idx);

  if (sz > _maxEvSize)
  {
    logging::critical("%s:\n  L1Accept of size %zd is too big for target buffer of size %zd\n",
                      __PRETTY_FUNCTION__, sz, _maxEvSize);
    abort();
    return -1;
  }

  if (_verbose >= VL_BATCH)
  {
    uint64_t pid    = ddg->pulseId();
    unsigned ctl    = ddg->control();
    uint32_t env    = ddg->env;
    void*    rmtAdx = (void*)link->rmtAdx(offset);
    printf("MebCtrb posts %9ld    monEvt [%5d]  @ "
           "%16p, ctl %02x, pid %014lx, env %08x, sz %6zd, MEB %2d @ %16p, data %08x\n",
           _eventCount, idx, ddg, ctl, pid, env, sz, link->id(), rmtAdx, data);
  }

  if (int rc = link->post(ddg, sz, offset, data) < 0)  return rc;

  ++_eventCount;

  return 0;
}

int MebContributor::post(const EbDgram* ddg)
{
  size_t              sz     = sizeof(*ddg) + ddg->xtc.sizeofPayload();
  TransitionId::Value tr     = ddg->service();
  uint64_t            offset = _bufRegSize + tr * _maxTrSize;

  if (sz > _maxTrSize)
  {
    logging::critical("%s:\n  %s transition of size %zd is too big for target buffer of size %zd\n",
                      __PRETTY_FUNCTION__, TransitionId::name(tr), sz, _maxTrSize);
    abort();
    return -1;
  }

  for (auto it = _links.begin(); it != _links.end(); ++it)
  {
    EbLfCltLink* link = *it;
    uint32_t     data = ImmData::value(ImmData::Transition, _id, tr);

    if (_verbose >= VL_BATCH)
    {
      uint64_t pid    = ddg->pulseId();
      unsigned ctl    = ddg->control();
      uint32_t env    = ddg->env;
      void*    rmtAdx = (void*)link->rmtAdx(offset);
      printf("MebCtrb posts %9ld      trId [%5d]  @ "
             "%16p, ctl %02x, pid %014lx, env %08x, sz %6zd, MEB %2d @ %16p - %16p, data %08x\n",
             _eventCount, tr, ddg, ctl, pid, env, sz, link->id(), rmtAdx, (char*)rmtAdx + sz, data);
    }

    if (int rc = link->post(ddg, sz, offset, data) < 0)  return rc;
  }

  return 0;
}
