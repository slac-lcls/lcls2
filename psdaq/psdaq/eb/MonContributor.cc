#include "psdaq/eb/MonContributor.hh"

#include "psdaq/eb/Endpoint.hh"
#include "psdaq/eb/EbLfClient.hh"

#include "psdaq/eb/utilities.hh"

#include "xtcdata/xtc/Dgram.hh"

#include <string.h>
#include <cstdint>
#include <string>
#include <unordered_map>

using namespace XtcData;
using namespace Pds::Eb;


MonContributor::MonContributor(const MonCtrbParams& prms) :
  _maxEvSize (roundUpSize(prms.maxEvSize)),
  _maxTrSize (prms.maxTrSize),
  _trOffset  (TransitionId::NumberOf),
  _region    (allocRegion(prms.maxEvents * _maxEvSize +
                          roundUpSize(TransitionId::NumberOf * prms.maxTrSize))),
  _transport (new EbLfClient()),
  _links     (),
  _id        (prms.id),
  _verbose   (prms.verbose),
  _eventCount(0)
{
  size_t regionSize = prms.maxEvents * _maxEvSize +
                      roundUpSize(TransitionId::NumberOf * prms.maxTrSize);

  if (_region == nullptr)
  {
    fprintf(stderr, "%s: No memory found for a input region of size %zd\n",
            __func__, regionSize);
    abort();
  }

  size_t offset = prms.maxEvents * _maxEvSize;
  for (unsigned i = 0; i < _trOffset.size(); ++i)
  {
    _trOffset[i] = offset;
    offset += _maxTrSize;
  }

  _initialize(__func__, prms.addrs, prms.ports, prms.id, regionSize);
}

MonContributor::~MonContributor()
{
  if (_transport)
  {
    for (EbLfLinkMap::iterator it  = _links.begin();
                               it != _links.end(); ++it)
    {
      _transport->shutdown(it->second);
    }
    _links.clear();
    delete _transport;
  }
  if (_region)  free(_region);
}

void MonContributor::_initialize(const char*                     who,
                                 const std::vector<std::string>& addrs,
                                 const std::vector<std::string>& ports,
                                 unsigned                        id,
                                 size_t                          regionSize)
{
  for (unsigned i = 0; i < addrs.size(); ++i)
  {
    const char*    addr = addrs[i].c_str();
    const char*    port = ports[i].c_str();
    EbLfLink*      link;
    const unsigned tmo(120000);         // Milliseconds
    if (_transport->connect(addr, port, tmo, &link))
    {
      fprintf(stderr, "%s: Error connecting to EbLfServer at %s:%s\n",
              who, addr, port);
      abort();
    }
    if (link->preparePoster(_region, regionSize, i, id, _verbose))
    {
      fprintf(stderr, "%s: Failed to prepare link to %s:%s\n",
              who, addr, port);
      abort();
    }
    _links[link->id()] = link;
    //_id2Idx[_links[i]->id()] = i;

    printf("%s: EbLfServer ID %d connected\n", who, link->id());
  }
}

int MonContributor::post(const Dgram* ddg, uint32_t destination)
{
  unsigned  dst    = ImmData::src(destination); //_id2Idx[ImmData::src(destination)];
  uint32_t  idx    = ImmData::idx(destination);
  size_t    sz     = sizeof(*ddg) + ddg->xtc.sizeofPayload();
  unsigned  offset = idx * _maxEvSize;
  EbLfLink* link   = _links[dst];
  uint32_t  data   = ImmData::buffer(_id /*link->index()*/, idx);

  if (sz > _maxEvSize)
  {
    fprintf(stderr, "%s: L1Accept of size %zd is too big for target buffer of size %zd\n",
            __PRETTY_FUNCTION__, sz, _maxEvSize);
    return -1;
  }

  if (_verbose)
  {
    uint64_t pid    = ddg->seq.pulseId().value();
    void*    rmtAdx = (void*)link->rmtAdx(offset);
    printf("MonCtrb posts %6ld   monEvt[%4d]    @ "
           "%16p, pid %014lx, sz %4zd to   Mon %d @ %16p, data %08x, phy %08x, dest %08x\n",
           _eventCount, idx, ddg, pid, sz, link->id(), rmtAdx, data, ddg->xtc.src.phy(), destination);
  }

  if (int rc = link->post(ddg, sz, offset, data) < 0)  return rc;

  ++_eventCount;

  return 0;
}

int MonContributor::post(const Dgram* ddg)
{
  size_t   sz      = sizeof(*ddg) + ddg->xtc.sizeofPayload();
  unsigned service = ddg->seq.service();
  uint64_t offset  = _trOffset[service];

  if (sz > _maxTrSize)
  {
    fprintf(stderr, "%s: %s transition of size %zd is too big for target buffer of size %zd\n",
            __PRETTY_FUNCTION__, TransitionId::name(ddg->seq.service()), sz, _maxTrSize);
    return -1;
  }

  for (EbLfLinkMap::iterator it  = _links.begin();
                             it != _links.end(); ++it)
  {
    EbLfLink* link = it->second;
    uint32_t  data = ImmData::transition(_id /*link->index()*/, service);

    if (_verbose)
    {
      uint64_t pid    = ddg->seq.pulseId().value();
      void*    rmtAdx = (void*)link->rmtAdx(offset);
      printf("MonCtrb posts %6ld     trId[%4d]    @ "
             "%16p, pid %014lx, sz %4zd to   Mon %d @ %16p, data %08x, phy %08x\n",
             _eventCount, service, ddg, pid, sz, link->id(), rmtAdx, data, ddg->xtc.src.phy());
    }

    if (int rc = link->post(ddg, sz, offset, data) < 0)  return rc;
  }

  return 0;
}
