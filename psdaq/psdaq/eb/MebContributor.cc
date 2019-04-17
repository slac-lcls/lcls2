#include "MebContributor.hh"

#include "Endpoint.hh"
#include "EbLfClient.hh"
#include "StatsMonitor.hh"

#include "utilities.hh"

#include "xtcdata/xtc/Dgram.hh"

#include <string.h>
#include <cstdint>
#include <string>

using namespace XtcData;
using namespace Pds::Eb;


MebContributor::MebContributor(const MebCtrbParams& prms, StatsMonitor& smon) :
  _maxEvSize (roundUpSize(prms.maxEvSize)),
  _maxTrSize (prms.maxTrSize),
  _trSize    (roundUpSize(TransitionId::NumberOf * _maxTrSize)),
  _transport (prms.verbose),
  _links     (),
  _id        (-1),
  _verbose   (prms.verbose),
  _eventCount(0)
{
  smon.metric("MCtbO_EvCt",  _eventCount,          StatsMonitor::SCALAR);
  smon.metric("MCtbO_TxPdg", _transport.pending(), StatsMonitor::SCALAR);
}

int MebContributor::connect(const MebCtrbParams& prms,
                            void*                region,
                            size_t               size)
{
  int    rc;
  size_t regSize = prms.maxEvents * _maxEvSize;

  _id         = prms.id;
  _eventCount = 0;
  _links.resize(prms.addrs.size());

  for (unsigned i = 0; i < prms.addrs.size(); ++i)
  {
    const char*    addr = prms.addrs[i].c_str();
    const char*    port = prms.ports[i].c_str();
    EbLfLink*      link;
    const unsigned tmo(120000);         // Milliseconds
    if ( (rc = _transport.connect(addr, port, tmo, &link)) )
    {
      fprintf(stderr, "%s:\n  Error connecting to MEB at %s:%s\n",
              __PRETTY_FUNCTION__, addr, port);
      return rc;
    }
    if ( (rc = link->preparePoster(prms.id, region, size, regSize)) )
    {
      fprintf(stderr, "%s:\n  Failed to prepare link with MEB at %s:%s\n",
              __PRETTY_FUNCTION__, addr, port);
      return rc;
    }
    _links[link->id()] = link;

    printf("Outbound link with MEB ID %d connected\n", link->id());
  }

  return 0;
}

void MebContributor::shutdown()
{
  for (auto it = _links.begin(); it != _links.end(); ++it)
  {
    _transport.shutdown(*it);
  }
  _links.clear();

  _id = -1;
}

int MebContributor::post(const Dgram* ddg, uint32_t destination)
{
  unsigned  dst    = ImmData::src(destination);
  uint32_t  idx    = ImmData::idx(destination);
  size_t    sz     = sizeof(*ddg) + ddg->xtc.sizeofPayload();
  unsigned  offset = _trSize + idx * _maxEvSize;
  EbLfLink* link   = _links[dst];
  uint32_t  data   = ImmData::value(ImmData::Buffer, _id, idx);

  if (sz > _maxEvSize)
  {
    fprintf(stderr, "%s:\n  L1Accept of size %zd is too big for target buffer of size %zd\n",
            __PRETTY_FUNCTION__, sz, _maxEvSize);
    return -1;
  }

  if (_verbose)
  {
    uint64_t pid    = ddg->seq.pulseId().value();
    unsigned ctl    = ddg->seq.pulseId().control();
    void*    rmtAdx = (void*)link->rmtAdx(offset);
    printf("MebCtrb posts %6ld       monEvt [%4d]  @ "
           "%16p, ctl %02d, pid %014lx, sz %4zd, MEB %2d @ %16p, data %08x\n",
           _eventCount, idx, ddg, ctl, pid, sz, link->id(), rmtAdx, data);
  }

  if (int rc = link->post(ddg, sz, offset, data) < 0)  return rc;

  ++_eventCount;

  return 0;
}

int MebContributor::post(const Dgram* ddg)
{
  size_t              sz  = sizeof(*ddg) + ddg->xtc.sizeofPayload();
  TransitionId::Value tr  = ddg->seq.service();
  uint64_t            ofs = tr * _maxTrSize;

  if (sz > _maxTrSize)
  {
    fprintf(stderr, "%s:\n  %s transition of size %zd is too big for target buffer of size %zd\n",
            __PRETTY_FUNCTION__, TransitionId::name(tr), sz, _maxTrSize);
    return -1;
  }

  for (auto it = _links.begin(); it != _links.end(); ++it)
  {
    EbLfLink* link = *it;
    uint32_t  data = ImmData::value(ImmData::Transition, _id, tr);

    if (_verbose)
    {
      uint64_t pid    = ddg->seq.pulseId().value();
      unsigned ctl    = ddg->seq.pulseId().control();
      void*    rmtAdx = (void*)link->rmtAdx(ofs);
      printf("MebCtrb posts %6ld         trId [%4d]  @ "
             "%16p, ctl %02x, pid %014lx, sz %4zd, MEB %2d @ %16p, data %08x\n",
             _eventCount, tr, ddg, ctl, pid, sz, link->id(), rmtAdx, data);
    }

    if (int rc = link->post(ddg, sz, ofs, data) < 0)  return rc;
  }

  return 0;
}
