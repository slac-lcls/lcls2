#include "EbLfLink.hh"

#include "Endpoint.hh"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <chrono>

using ms_t = std::chrono::milliseconds;

using namespace Pds;
using namespace Pds::Fabrics;
using namespace Pds::Eb;

static const unsigned MaxLinks = 64;    // Maximum supported


EbLfLink::EbLfLink(Endpoint* ep) :
  _ep(ep),
  _mr(nullptr),
  _ra(),
  _rxDepth(0),
  _rOuts(0),
  _idx(-1),
  _id(-1),
  _region(nullptr),
  _verbose(0)
{
}

EbLfLink::EbLfLink(Endpoint* ep, int rxDepth) :
  _ep(ep),
  _mr(nullptr),
  _ra(),
  _rxDepth(rxDepth),
  _rOuts(0),
  _idx(-1),
  _id(-1),
  _region(nullptr),
  _verbose(0)
{
}

EbLfLink::~EbLfLink()
{
  if (_region)  delete [] _region;
}

int EbLfLink::preparePender(unsigned idx,
                            unsigned id,
                            unsigned verbose,
                            void*    ctx)
{
  int    rc;
  size_t size = sizeof(RemoteAddress);

  _verbose = verbose;

  if (!_region)  _region = new char[size];
  else           return -1;
  if (!_region)  return ENOMEM;

  if ( (rc = setupMr(_region, size)) ) return rc;

  if ( (rc = recvId()  ) )             return rc;
  if ( (rc = sendId(idx, id)) )        return rc;

  if ( (rc = syncLclMr()) )            return rc;

  postCompRecv(ctx);

  return rc;
}

int EbLfLink::preparePender(void*    region,
                            size_t   size,
                            unsigned idx,
                            unsigned id,
                            unsigned verbose,
                            void*    ctx)
{
  int rc;

  _verbose = verbose;

  if ( (rc = setupMr(region, size)) )  return rc;

  if ( (rc = recvId()  ) )             return rc;
  if ( (rc = sendId(idx, id)) )        return rc;

  if ( (rc = syncLclMr()) )            return rc;

  postCompRecv(ctx);

  return rc;
}

int EbLfLink::preparePoster(unsigned idx,
                            unsigned id,
                            unsigned verbose)
{
  int    rc;
  size_t size = sizeof(RemoteAddress);

  _verbose = verbose;

  if (!_region)  _region = new char[size];
  else           return -1;
  if (!_region)  return ENOMEM;

  if ( (rc = setupMr(_region, size)) ) return rc;

  if ( (rc = sendId(idx, id)) )        return rc;
  if ( (rc = recvId()  ) )             return rc;

  if ( (rc = syncRmtMr(size)) )        return rc;

  return 0;
}

int EbLfLink::preparePoster(void*    region,
                            size_t   size,
                            unsigned idx,
                            unsigned id,
                            unsigned verbose)
{
  int rc;

  _verbose = verbose;

  if ( (rc = setupMr(region, size)) )  return rc;

  if ( (rc = sendId(idx, id)) )        return rc;
  if ( (rc = recvId()  ) )             return rc;

  if ( (rc = syncRmtMr(size)) )        return rc;

  return 0;
}

int EbLfLink::setupMr(void*  region,
                      size_t size)
{
  Fabric* fab = _ep->fabric();

  _mr = fab->register_memory(region, size);
  if (!_mr)
  {
    fprintf(stderr, "%s: Failed to register memory region @ %p, size %zu: %s\n",
            __PRETTY_FUNCTION__, region, size, fab->error());
    return fab->error_num();
  }

  if (_verbose)
  {
    printf("Registered      memory region: %10p : %10p, size %zd\n",
           region, (char*)region + size, size);
  }

  return 0;
}

int EbLfLink::recvId()
{
  ssize_t rc;
  void*   buf = _mr->start();

  if ((rc = _ep->recv_sync(buf, sizeof(unsigned), _mr)) < 0)
  {
    fprintf(stderr, "%s: Failed receiving ID from peer: %s\n",
            __PRETTY_FUNCTION__, _ep->error());
    return rc;
  }
  _id  = (*(unsigned*)buf) >> 8;
  _idx = (*(unsigned*)buf) & (MaxLinks - 1);

  return 0;
}

int EbLfLink::sendId(unsigned idx,
                     unsigned id)
{
  ssize_t      rc;
  void*        buf = _mr->start();

  LocalAddress adx(buf, sizeof(unsigned), _mr);
  LocalIOVec   iov(&adx, 1);
  Message      msg(&iov, 0, NULL, 0);

  *(unsigned*)buf = (id << 8) | idx;
  if ((rc = _ep->sendmsg_sync(&msg, FI_TRANSMIT_COMPLETE | FI_COMPLETION)) < 0)
  {
    fprintf(stderr, "%s: Failed sending our ID to peer: %s\n",
            __PRETTY_FUNCTION__, _ep->error());
    return rc;
  }

  return 0;
}

int EbLfLink::syncLclMr()
{
  void*  buf = _mr->start();
  size_t len = _mr->length();
  _ra = RemoteAddress(_mr->rkey(), (uint64_t)buf, len);

  LocalAddress adx(buf, sizeof(_ra), _mr);
  LocalIOVec   iov(&adx, 1);
  Message      msg(&iov, 0, NULL, 0);
  ssize_t      rc;

  memcpy(buf, &_ra, sizeof(_ra));
  if ((rc = _ep->sendmsg_sync(&msg, FI_COMPLETION)) < 0)
  {
    fprintf(stderr, "%s: Failed sending local memory specs to ID %d: %s\n",
            __PRETTY_FUNCTION__, _id, _ep->error());
    return rc;
  }

  if (_verbose)
  {
    printf("Sent     local  memory region: %10p : %10p, size %zd\n",
           buf, (char*)buf + len, len);
  }

  return 0;
}

int EbLfLink::syncRmtMr(size_t size)
{
  ssize_t  rc;
  if ((rc = _ep->recv_sync(_mr->start(), sizeof(_ra), _mr)) < 0)
  {
    fprintf(stderr, "%s: Failed receiving remote region specs from ID %d: %s\n",
            __PRETTY_FUNCTION__, _id, _ep->error());
    return rc;
  }

  memcpy(&_ra, _mr->start(), sizeof(_ra));

  if (size > _ra.extent)
  {
    fprintf(stderr, "%s: Remote region size (%lu) is less than required (%lu)\n",
            __PRETTY_FUNCTION__, _ra.extent, size);
    return -1;
  }

  if (_verbose)
  {
    printf("Received remote memory region: %10p : %10p, size %zd\n",
           (void*)_ra.addr, (void*)(_ra.addr + _ra.extent), _ra.extent);
  }

  return 0;
}

int EbLfLink::postCompRecv(void* ctx)
{
  if (--_rOuts <= 1)
  {
    unsigned count = _rxDepth - _rOuts;
    _rOuts += _postCompRecv(count, ctx);
    if (_rOuts < _rxDepth)
    {
      fprintf(stderr, "%s: Failed to post CQ buffers: %d of %d available\n",
              __PRETTY_FUNCTION__, _rOuts, _rxDepth);
      return _rxDepth - _rOuts;
    }
  }

  return 0;
}

int EbLfLink::_postCompRecv(unsigned count, void* ctx)
{
  unsigned i;

  for (i = 0; i < count; ++i)
  {
    ssize_t rc;
    if ((rc = _ep->recv_comp_data(ctx)) < 0)
    {
      if (rc != -FI_EAGAIN)
        fprintf(stderr, "%s: Failed to post a CQ buffer: %s\n",
                __PRETTY_FUNCTION__, _ep->error());
      break;
    }
  }

  return i;
}

int EbLfLink::post(const void* buf,
                   size_t      len,
                   uint64_t    offset,
                   uint64_t    immData,
                   void*       ctx)
{
  RemoteAddress ra(_ra.rkey, _ra.addr + offset, len);
  ssize_t       rc;

  while ((rc = _ep->write_data(buf, len, &ra, ctx, immData, _mr)) < 0)
  {
    if (rc != -FI_EAGAIN)
    {
      fprintf(stderr, "%s: write_data to ID %d failed: %s\n",
              __PRETTY_FUNCTION__, _id, _ep->error());
      break;
    }

    fi_cq_data_entry cqEntry;
    const ssize_t    maxCnt = 1;
    CompletionQueue* cq     = _ep->txcq();
    rc = cq->comp(&cqEntry, maxCnt);
    if ((rc != -FI_EAGAIN) && (rc != maxCnt)) // EAGAIN means no completions available
    {
      fprintf(stderr, "%s: Error reading TX CQ: %s\n",
              __PRETTY_FUNCTION__, cq->error());
      break;
    }
  }

  return rc;
}

int EbLfLink::post(const void* buf,
                   size_t      len,
                   uint64_t    immData)
{
  if (ssize_t rc = _ep->inject_data(buf, len, immData) < 0)
  {
    fprintf(stderr, "%s: inject_data failed: %s\n",
            __PRETTY_FUNCTION__, _ep->error());
    return rc;
  }

  return 0;
}
