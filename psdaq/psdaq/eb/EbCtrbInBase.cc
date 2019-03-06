#include "EbCtrbInBase.hh"

#include "psdaq/eb/Endpoint.hh"
#include "psdaq/eb/EbLfServer.hh"
#include "psdaq/eb/Batch.hh"
#include "psdaq/eb/TebContributor.hh"
#include "psdaq/eb/utilities.hh"

#include "xtcdata/xtc/Dgram.hh"

#include <string>
#include <bitset>


using namespace XtcData;
using namespace Pds::Eb;


EbCtrbInBase::EbCtrbInBase(const TebCtrbParams& prms) :
  _transport   (prms.verbose),
  _links       (),
  _maxBatchSize(0),
  _prms        (prms),
  _region      (nullptr)
{
}

int EbCtrbInBase::connect(const TebCtrbParams& prms)
{
  int rc;

  unsigned numEbs = std::bitset<64>(prms.builders).count();
  _links.resize(numEbs);

  if ( (rc = _transport.initialize(prms.ifAddr, prms.port, numEbs)) )
  {
    fprintf(stderr, "%s:\n  Failed to initialize EbLfServer\n",
            __PRETTY_FUNCTION__);
    return rc;
  }

  size_t size = 0;

  // Since each EB handles a specific batch, one region can be shared by all
  for (unsigned i = 0; i < numEbs; ++i)
  {
    EbLfLink*      link;
    const unsigned tmo(120000);         // Milliseconds
    if ( (rc = _transport.connect(&link, tmo)) )
    {
      fprintf(stderr, "%s:\n  Error connecting to TEB %d\n",
              __PRETTY_FUNCTION__, i);
      return rc;
    }

    size_t regSize;
    if ( (rc = link->preparePender(prms.id, &regSize)) )
    {
      fprintf(stderr, "%s:\n  Failed to prepare link with TEB %d\n",
              __PRETTY_FUNCTION__, i);
      return rc;
    }
    _links[link->id()] = link;

    if (!size)
    {
      size          = regSize;
      _maxBatchSize = regSize / prms.maxBatches;

      _region = allocRegion(regSize);
      if (_region == nullptr)
      {
        fprintf(stderr, "%s:\n  No memory found for a Result MR of size %zd\n",
                __PRETTY_FUNCTION__, regSize);
        return ENOMEM;
      }
    }
    else if (regSize != size)
    {
      fprintf(stderr, "%s:\n  Error: Result MR size (%zd) cannot differ between TEBs "
              "(%zd from Id %d)\n", __PRETTY_FUNCTION__, size, regSize, link->id());
      return -1;
    }

    if ( (rc = link->setupMr(_region, regSize)) )
    {
      char* region = static_cast<char*>(_region);
      fprintf(stderr, "%s:\n  Failed to set up Result MR for TEB ID %d, %p:%p, size %zd\n",
              __PRETTY_FUNCTION__, link->id(), region, region + regSize, regSize);
      return rc;
    }
    link->postCompRecv();

    printf("Inbound link with TEB ID %d connected\n", link->id());
  }

  return 0;
}

void EbCtrbInBase::shutdown()
{
  for (auto it = _links.begin(); it != _links.end(); ++it)
  {
    _transport.shutdown(*it);
  }
  _links.clear();
  _transport.shutdown();

  if (_region)  free(_region);
  _region = nullptr;
}

int EbCtrbInBase::process(BatchManager& batMan)
{
  // Pend for a result datagram (batch) and process it.
  uint64_t  data;
  int       rc;
  const int tmo = 5000;                 // milliseconds
  auto t0(std::chrono::steady_clock::now());
  if ( (rc = _transport.pend(&data, tmo)) < 0)  return rc;
  auto t1(std::chrono::steady_clock::now());

  unsigned     src = ImmData::src(data);
  unsigned     idx = ImmData::idx(data);
  EbLfLink*    lnk = _links[src];
  const Dgram* bdg = (const Dgram*)(lnk->lclAdx(idx * _maxBatchSize));
  lnk->postCompRecv();

  if (_prms.verbose)
  {
    static unsigned cnt = 0;
    uint64_t        pid = bdg->seq.pulseId().value();
    unsigned        ctl = bdg->seq.pulseId().control();
    size_t          sz  = sizeof(*bdg) + bdg->xtc.sizeofPayload();
    printf("CtrbIn  rcvd        %6d result  [%4d] @ "
           "%16p, ctl %02x, pid %014lx, sz %4zd, src %2d\n",
           cnt++, idx, bdg, ctl, pid, sz, lnk->id());
  }

  Dgram const* result = bdg;
  const Batch* inputs = batMan.batch(idx);
  //printf("        data %08lx           idx %4d rPid %014lx iPid %014lx               id %2d svc %d\n",
  //       data, idx, result->seq.pulseId().value(), inputs->datagram()->seq.pulseId().value(),
  //       lnk->id(), result->seq.service());
  unsigned i = 0;
  while (true)
  {
    process(result, inputs->appParm(i++));

    if (!result->seq.isBatch())  break; // Last event in batch does not have Batch bit set

    result = reinterpret_cast<const Dgram*>(result->xtc.next());
  }

  batMan.release(inputs);

  return 0;
}
