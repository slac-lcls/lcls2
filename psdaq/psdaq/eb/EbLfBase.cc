#include "EbLfBase.hh"

#include "Endpoint.hh"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <chrono>

typedef std::chrono::milliseconds ms_t;

using namespace Pds;
using namespace Pds::Fabrics;
using namespace Pds::Eb;

static const size_t scratch_size = sizeof(Fabrics::RemoteAddress);


EbLfStats::EbLfStats(unsigned nPeers) :
  _postCnt(0),
  _repostCnt(0),
  _repostMax(0),
  _pendCnt(0),
  _pendTmoCnt(0),
  _rependCnt(0),
  _rependMax(0)
{
}

EbLfStats::~EbLfStats()
{
}

void EbLfStats::clear()
{
  _postCnt      = 0;
  _repostCnt    = 0;
  _repostMax    = 0;
  _pendCnt      = 0;
  _rependCnt    = 0;
  _rependMax    = 0;
}

void EbLfStats::dump()
{
  if (_postCnt)
  {
    printf("post: count %8ld, reposts %8ld (max %8ld)\n",
           _postCnt, _repostCnt, _repostMax);
  }
  if (_pendCnt)
  {
    printf("pend: count %8ld, repends %8ld (max %8ld), timeouts %8ld\n",
           _pendCnt, _rependCnt, _rependMax, _pendTmoCnt);

  }
}


EbLfBase::EbLfBase(unsigned nPeers) :
  _ep(nPeers),
  _mr(nPeers),
  _ra(nPeers),
  _txcq(nPeers),
  _rxcq(0),
  _rxDepth(0),
  _rOuts(nPeers),
  _id(nPeers),
  _mappedId(nullptr),
  _stats(nPeers)
{
}

EbLfBase::~EbLfBase()
{
  if (_mappedId)  delete [] _mappedId;
}

int EbLfBase::_setupMr(Endpoint*      ep,
                       void*          region,
                       size_t         size,
                       MemoryRegion*& mr)
{
  Fabric* fab = ep->fabric();

  mr = fab->register_memory(region, size);
  if (!mr)
  {
    fprintf(stderr, "Failed to register memory region @ %p, size %zu: %s\n",
            region, size, fab->error());
    return fab->error_num();
  }

  printf("Registered      memory region: %10p : %10p, size %zd\n",
         region, (char*)region + size, size);

  return 0;
}

int EbLfBase::_syncLclMr(Endpoint*      ep,
                         MemoryRegion*  mr,
                         RemoteAddress& ra)
{
  ssize_t      rc;
  void*        buf = mr->start();
  size_t       len = mr->length();
  ra = RemoteAddress(mr->rkey(), (uint64_t)buf, len);

  LocalAddress adx(buf, sizeof(ra), mr);
  LocalIOVec   iov(&adx, 1);
  Message      msg(&iov, 0, NULL, 0);

  memcpy(buf, &ra, sizeof(ra));
  if ((rc = ep->sendmsg_sync(&msg, FI_COMPLETION)) < 0)
  {
    fprintf(stderr, "Failed sending local memory specs to peer: %s\n",
            ep->error());
    return rc;
  }

  printf("Sent     local  memory region: %10p : %10p, size %zd\n",
         buf, (char*)buf + len, len);

  return 0;
}

int EbLfBase::_syncRmtMr(Endpoint*      ep,
                         MemoryRegion*  mr,
                         RemoteAddress& ra,
                         size_t         size)
{
  ssize_t  rc;
  if ((rc = ep->recv_sync(mr->start(), sizeof(ra), mr)) < 0)
  {
    fprintf(stderr, "Failed receiving remote region specs from peer: %s\n",
            ep->error());
    return rc;
  }

  memcpy(&ra, mr->start(), sizeof(ra));

  if (size > ra.extent)
  {
    fprintf(stderr, "Remote region size (%lu) is less than required (%lu)\n",
            ra.extent, size);
    return -1;
  }

  printf("Received remote memory region: %10p : %10p, size %zd\n",
         (void*)ra.addr, (void*)(ra.addr + ra.extent), ra.extent);

  return 0;
}

void EbLfBase::_mapIds(unsigned nPeers)
{
  unsigned idMax = 0;
  for (unsigned i = 0; i < nPeers; ++i)
    if (_id[i] > idMax) idMax = _id[i];

  _mappedId = new unsigned[idMax + 1];
  assert(_mappedId);

  for (unsigned i = 0; i < idMax + 1; ++i)
    _mappedId[i] = -1;

  for (unsigned i = 0; i < nPeers; ++i)
    _mappedId[_id[i]] = i;
}

const EbLfStats& EbLfBase::stats() const
{
  return _stats;
}

void* EbLfBase::lclAdx(unsigned src, uint64_t offset) const
{
  unsigned idx = _mappedId[src];
  if (idx == -1u)
  {
    fprintf(stderr, "%s: Invalid ID: %d\n", __PRETTY_FUNCTION__, src);
    return nullptr;
  }

  return (char*)_mr[_mappedId[src]]->start() + offset;
}

uintptr_t EbLfBase::rmtAdx(unsigned dst, uint64_t offset) const
{
  unsigned idx = _mappedId[dst];
  if (idx == -1u)
  {
    fprintf(stderr, "%s: Invalid ID: %d\n", __PRETTY_FUNCTION__, dst);
    return 0;
  }

  return _ra[_mappedId[dst]].addr + offset;
}

int EbLfBase::postCompRecv(unsigned dst, void* ctx)
{
  unsigned idx = _mappedId[dst];
  if (idx == -1u)
  {
    fprintf(stderr, "%s: Invalid ID: %d\n", __PRETTY_FUNCTION__, dst);
    return -1;
  }

  if (--_rOuts[idx] <= 1)
  {
    unsigned count = _rxDepth - _rOuts[idx];
    _rOuts[idx] += _postCompRecv(_ep[idx], count, ctx);
    if (_rOuts[idx] < _rxDepth)
    {
      fprintf(stderr, "Failed to post all %d receives for id %d: %d\n",
              count, _id[idx], _rOuts[idx]);
    }
  }

  return _rOuts[idx];
}

int EbLfBase::_postCompRecv(Endpoint* ep, unsigned count, void* ctx)
{
  unsigned i;

  for (i = 0; i < count; ++i)
  {
    ssize_t rc;
    if ((rc = ep->recv_comp_data(ctx)) < 0)
    {
      if (rc != -FI_EAGAIN)
        fprintf(stderr, "Failed to post a CQ buffer: %s\n", ep->error());
      break;
    }
  }

  return i;
}

int EbLfBase::_tryCq(fi_cq_data_entry* cqEntry)
{
  const int maxCnt = 1;
  //const int tmo    = 5000;              // milliseconds
  //ssize_t rc = _rxcq->comp_wait(cqEntry, maxCnt, tmo); // Waiting favors throughput
  ssize_t rc = _rxcq->comp(cqEntry, maxCnt);           // Polling favors latency
  if (rc == maxCnt)
  {
    const uint64_t flags = FI_REMOTE_WRITE | FI_REMOTE_CQ_DATA;
    if ((cqEntry->flags & flags) == flags)
    {
      //fprintf(stderr, "Expected   CQ entry: count %zd, got flags %016lx vs %016lx, data = %08lx\n",
      //        rc, cqEntry->flags, flags, cqEntry->data);
      //fprintf(stderr, "                     ctx   %p, len %zd, buf %p\n",
      //        cqEntry->op_context, cqEntry->len, cqEntry->buf);

      return 0;
    }

    fprintf(stderr, "Unexpected CQ entry: count %zd, got flags %016lx vs %016lx\n",
            rc, cqEntry->flags, flags);
    fprintf(stderr, "                     ctx   %p, len %zd, buf %p\n",
            cqEntry->op_context, cqEntry->len, cqEntry->buf);
    return -FI_EAGAIN;
  }
  else
  {
    if (rc != -FI_EAGAIN)
    {
      static int _errno = 0;
      if (rc != _errno)
      {
        fprintf(stderr, "Error reading RX completion queue: %s\n",
                _rxcq->error());
        _errno = rc;
      }
      return rc;
    }
  }

  return -FI_EAGAIN;
}

int EbLfBase::pend(fi_cq_data_entry* cqEntry)
{
  int      rc;
  uint64_t rependCnt = 0;
  ++_stats._pendCnt;

  auto t0 = std::chrono::steady_clock::now();
  while ((rc = _tryCq(cqEntry)) == -FI_EAGAIN)
  {
    auto t1 = std::chrono::steady_clock::now();

    const int tmo = 5000;               // milliseconds
    if (std::chrono::duration_cast<ms_t>(t1 - t0).count() > tmo)
    {
      ++_stats._pendTmoCnt;
      return -FI_ETIMEDOUT;
    }

    ++rependCnt;
  }

  if (rependCnt > _stats._rependMax)
    _stats._rependMax = rependCnt;
  _stats._rependCnt += rependCnt;

  return rc;
}

int EbLfBase::post(unsigned    dst,
                   const void* buf,
                   size_t      len,
                   uint64_t    offset,
                   uint64_t    immData,
                   void*       ctx)
{
  uint64_t repostCnt = 0;
  ++_stats._postCnt;

  ssize_t          rc;
  unsigned         idx = _mappedId[dst];
  if (idx == -1u)
  {
    fprintf(stderr, "%s: Invalid ID: %d\n", __PRETTY_FUNCTION__, dst);
    return -FI_EINVAL;
  }

  Endpoint*        ep  = _ep[idx];
  RemoteAddress    ra   (_ra[idx].rkey, _ra[idx].addr + offset, len);
  MemoryRegion*    mr  = _mr[idx];
  CompletionQueue* cq  = ep->txcq();    // Same as _txcq[idx]

  while ((rc = ep->write_data(buf, len, &ra, ctx, immData, mr)) < 0)
  {
    if (rc != -FI_EAGAIN)
    {
      fprintf(stderr, "write_data failed: %s\n", ep->error());
      break;
    }
    ++repostCnt;

    fi_cq_data_entry cqEntry;
    const ssize_t    maxCnt = 1;
    rc = cq->comp(&cqEntry, maxCnt);
    if ((rc != -FI_EAGAIN) && (rc != maxCnt)) // EAGAIN means no completions available
    {
      fprintf(stderr, "Error reading TX completing queue %u: %s\n",
              idx, cq->error());
      break;
    }
  }

  if (repostCnt > _stats._repostMax)
    _stats._repostMax = repostCnt;
  _stats._repostCnt += repostCnt;

  return rc;
}
