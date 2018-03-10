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
  _rependMax(0),
  _rmtWrCnt(nPeers),
  _compAgnCnt(nPeers)
{
  for (unsigned i = 0; i < nPeers; ++i)
  {
    _rmtWrCnt[i]    = 0;
    _compAgnCnt[i]  = 0;
  }
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

  for (unsigned i = 0; i < _rmtWrCnt.size(); ++i)
  {
    _rmtWrCnt[i]    = 0;
    _compAgnCnt[i]  = 0;
  }
}

static void prtVec(const char* item, const std::vector<uint64_t>& stat)
{
  unsigned i;

  printf("%s:\n", item);
  for (i = 0; i < stat.size(); ++i)
  {
    printf(" %8ld", stat[i]);
    if ((i % 8) == 7)  printf("\n");
  }
  if ((--i % 8) != 7)  printf("\n");
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

    prtVec("rmtWrCnt",    _rmtWrCnt);
    prtVec("compAgnCnt",  _compAgnCnt);
  }
}


EbLfBase::EbLfBase(unsigned nPeers) :
  _ep(nPeers),
  _lMr(nPeers),
  _rMr(nPeers),
  _ra(nPeers),
  _cqPoller(nullptr),
  _base(nullptr),
  _rxDepth(0),
  _rOuts(nPeers),
  _id(nPeers),
  _mappedId(nullptr),
  _stats(nPeers),
  _iSrc(0)
{
}

EbLfBase::~EbLfBase()
{
  unsigned nEp = _ep.size();
  for (unsigned i = 0; i < nEp; ++i)
    if (_cqPoller && _ep[i])  _cqPoller->del(_ep[i]);
  if (_cqPoller)  delete _cqPoller;

  if (_base)      free(_base);
  if (_mappedId)  delete [] _mappedId;
}

const char* EbLfBase::base() const
{
  return _base;
}

int EbLfBase::prepareLclMr(size_t lclSize, PeerSharing shared)
{
  size_t   alignment = sysconf(_SC_PAGESIZE);
  assert(lclSize & (alignment - 1) == 0);
  unsigned nClients  = _ep.size();
  size_t   size      = (shared == PEERS_SHARE_BUFFERS ? 1 : nClients) * lclSize;
  void*    lclMem    = nullptr;
  int      ret       = posix_memalign(&lclMem, alignment, size);
  if (ret)
  {
    perror("posix_memalign");
    return ret;
  }
  if (lclMem == nullptr)
  {
    fprintf(stderr, "No memory found for a region of size %zd\n", lclSize);
    return -1;
  }
  _base = (char*)lclMem;

  char*    region = _base;
  for (unsigned i = 0; i < _ep.size(); ++i)
  {
    Endpoint*     ep  = _ep[i];
    Fabric*       fab = ep->fabric();
    MemoryRegion* mr  = fab->register_memory(region, lclSize);
    if (!mr)
    {
      fprintf(stderr, "Failed to register memory region @ %p, size %zu: %s\n",
              region, lclSize, fab->error());
      return fab->error_num();
    }
    _lMr[i] = mr;

    printf("Sink   memory region[%2d]: %10p : %10p, size %zd\n",
           i, region, (char*)region + lclSize, lclSize);

    ssize_t       rc;
    RemoteAddress ra(mr->rkey(), (uint64_t)mr->start(), mr->length());
    memcpy(mr->start(), &ra, sizeof(ra));

    if ((rc = ep->send_sync(mr->start(), sizeof(ra), mr)) < 0)
    {
      fprintf(stderr, "Failed sending local memory specs to peer: %s\n",
              ep->error());
      return rc;
    }

    if (shared == PER_PEER_BUFFERS)  region += lclSize;
  }

  return ret;
}

//#include <sys/mman.h>

int EbLfBase::prepareRmtMr(void* region, size_t rmtSize)
{
  // No obvious effect:
  //int ret = mlock(region, rmtSize);
  //if (ret) perror ("mlock");

  for (unsigned i = 0; i < _ep.size(); ++i)
  {
    Endpoint*     ep  = _ep[i];
    Fabric*       fab = ep->fabric();
    MemoryRegion* mr  = fab->register_memory(region, rmtSize);
    if (!mr)
    {
      fprintf(stderr, "Failed to register memory region @ %p, size %zu: %s\n",
              region, rmtSize, fab->error());
      return fab->error_num();
    }
    _rMr[i] = mr;

    printf("Source memory region[%2d]: %10p : %10p, size %zd\n",
           i, region, (char*)region + rmtSize, rmtSize);

    ssize_t        rc;
    RemoteAddress& ra = _ra[i];
    if ((rc = ep->recv_sync(mr->start(), sizeof(ra), mr)) < 0)
    {
      fprintf(stderr, "Failed receiving remote region specs from peer: %s\n",
              ep->error());
      return rc;
    }

    memcpy(&ra, mr->start(), sizeof(ra));

    if (rmtSize > ra.extent)
    {
      fprintf(stderr, "Remote region size (%lu) is less than required (%lu)\n",
              ra.extent, rmtSize);
      return -1;
    }

    printf("Remote memory region[%2d]: %10p : %10p, size %zd\n",
           i, (void*)ra.addr, (void*)(ra.addr + ra.extent), ra.extent);

    if ((rc = ep->recv_comp_data()) < 0)
    {
      if (rc != -FI_EAGAIN)
        fprintf(stderr, "recv_comp_data() error for index %d: %s\n",
                i, ep->error());
    }
  }

  return 0;
}

void EbLfBase::_mapIds(unsigned nPeers)
{
  unsigned idMax = 0;
  for (unsigned i = 0; i < nPeers; ++i)
    if (_id[i] > idMax) idMax = _id[i];

  _mappedId = new unsigned[idMax + 1];
  assert(_mappedId);

  for (unsigned i = 0; i < nPeers; ++i)
    _mappedId[_id[i]] = i;
}

const EbLfStats& EbLfBase::stats() const
{
  return _stats;
}

void* EbLfBase::lclAdx(unsigned src, uint64_t offset) const
{
  return (char*)_lMr[_mappedId[src]]->start() + offset;
}

uintptr_t EbLfBase::rmtAdx(unsigned dst, uint64_t offset) const
{
  return _ra[_mappedId[dst]].addr + offset;
}

int EbLfBase::postCompRecv(unsigned dst, unsigned count, void* ctx)
{
  Endpoint* ep = _ep[_mappedId[dst]];
  unsigned  i;

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
  // Cycle through all sources to find which one has data
  for (unsigned i = 0; i < _ep.size(); ++i)
  {
    unsigned iSrc = _iSrc++;
    if (_iSrc == _ep.size())  _iSrc = 0;

    Endpoint*& ep = _ep[iSrc];
    if (!ep)  continue;

    ssize_t   rc;
    const int maxCnt = 1;
    if ((rc = ep->comp(ep->rxcq(), cqEntry, maxCnt)) == maxCnt)
    {
      const uint64_t flags = FI_REMOTE_WRITE | FI_REMOTE_CQ_DATA;

      if ((rc = ep->recv_comp_data()) < 0)
      {
        if (rc != -FI_EAGAIN)
          fprintf(stderr, "tryCq: recv_comp_data() error: %s\n", ep->error());
      }

      if ((cqEntry->flags & flags) == flags)
      {
        ++_stats._rmtWrCnt[iSrc];

        //fprintf(stderr, "Expected   completion queue entry of peer %u: count %d, got flags %016lx vs %016lx, data = %08lx\n",
        //        _id[iSrc], compCnt, cqEntry->flags, flags, cqEntry->data);

        //*data = (char*)_lMr[iSrc]->start() + cqEntry->data; // imm_data is only 32 bits for verbs!
        return 0;
      }

      fprintf(stderr, "Unexpected completion queue entry of peer %u: count %ld, got flags %016lx vs %016lx\n",
              _id[iSrc], rc, cqEntry->flags, flags);
      return 1;
    }
    else
    {
      if (rc != -FI_EAGAIN)
      {
        static int _errno = 0;
        if (rc != _errno)
        {
          fprintf(stderr, "Error reading completion queue of peer %u: %s\n",
                  _id[iSrc], ep->error());
          _errno = rc;
        }
        return rc;
      }
    }

    ++_stats._compAgnCnt[iSrc];
  }

  return -FI_EAGAIN;
}

int EbLfBase::pend(fi_cq_data_entry* cqEntry)
{
  int      rc;
  uint64_t rependCnt  = 0;
  ++_stats._pendCnt;
  auto t0 = std::chrono::steady_clock::now();

  while ((rc = _tryCq(cqEntry)) < 0)
  {
    if (rc != -FI_EAGAIN)  return rc;

    const int tmo = 5000;               // milliseconds
    auto t1 = std::chrono::steady_clock::now();

    if (std::chrono::duration_cast<ms_t>(t1 - t0).count() > tmo)
      return -FI_ETIMEDOUT;

    ++rependCnt;
  }

  if (rependCnt > _stats._rependMax)
    _stats._rependMax = rependCnt;
  _stats._rependCnt += rependCnt;

  return rc;
}

int EbLfBase::pendW(fi_cq_data_entry* cqEntry)
{
  int      rc;
  uint64_t rependCnt  = 0;
  ++_stats._pendCnt;

  while ((rc = _tryCq(cqEntry)) < 0)
  {
    if (rc != -FI_EAGAIN)  return rc;

    const int tmo = 5000;               // milliseconds
    rc = _cqPoller->poll(tmo);
    if ((rc <= 0) && (rc != -FI_EAGAIN))
    {
      if (rc == 0)
      {
        ++_stats._pendTmoCnt;
      }
      else
      {
        fprintf(stderr, "Error polling completion queues: %s\n",
                _cqPoller->error());
      }
      return rc;
    }
    ++rependCnt;
  }

  if (rependCnt > _stats._rependMax)
    _stats._rependMax = rependCnt;
  _stats._rependCnt += rependCnt;

  return rc;
}

#if 0
static int pollForComp(Endpoint* ep, unsigned idx)
{
  ssize_t          rc;
  fi_cq_data_entry cqEntry;
  const ssize_t    maxCnt = 1;
  auto t0 = std::chrono::steady_clock::now();

  while (1)
  {
    if ((rc = ep->recv_comp_data()) < 0)
    {
      if (rc != -FI_EAGAIN)
        fprintf(stderr, "pollForComp: recv_comp_data() error: %s\n", ep->error());
    }

    if ((rc = ep->comp(ep->txcq(), &cqEntry, maxCnt)) == maxCnt)  return 0;

    if (rc != -FI_EAGAIN)
    {
      fprintf(stderr, "Error completing post to peer %u: %s\n",
              idx, ep->error());
      return rc;
    }

    const int tmo = 5000;               // milliseconds
    auto t1 = std::chrono::steady_clock::now();

    if (std::chrono::duration_cast<ms_t>(t1 - t0).count() > tmo)
      return -FI_ETIMEDOUT;
  }
}
#endif

int EbLfBase::post(const void* buf,
                   size_t      len,
                   unsigned    dst,
                   uint64_t    immData,
                   void*       ctx)
{
  uint64_t repostCnt = 0;
  ++_stats._postCnt;

  unsigned      idx = _mappedId[dst];
  Endpoint*     ep  = _ep[idx];
  uintptr_t     os  = (const char*)buf - (const char*)_rMr[idx]->start();
  RemoteAddress ra   (_ra[idx].rkey, _ra[idx].addr + os, len);
  MemoryRegion* mr  = _rMr[idx];
  ssize_t       rc;
  do
  {
    if ((rc = ep->write_data(buf, len, &ra, ctx, immData, mr)) < 0)
    {
      if (rc != -FI_EAGAIN)
      {
        fprintf(stderr, "write_data failed: %s\n", ep->error());
        --repostCnt;
      }
      ++repostCnt;
    }

    // It appears that the CQ must always be read, not just after EAGAIN
    //if ( (rc = pollForComp(ep, idx)) )  return rc;

    ssize_t          ret;
    fi_cq_data_entry cqEntry;
    const ssize_t    maxCnt = 1;
    if ((ret = ep->comp(ep->txcq(), &cqEntry, maxCnt)) == maxCnt)
    {
      if ((ret = ep->recv_comp_data()) < 0)
      {
        if (ret != -FI_EAGAIN)
          fprintf(stderr, "pollForComp: recv_comp_data() error: %s\n", ep->error());
      }
    }
    else if (ret != -FI_EAGAIN)
    {
      fprintf(stderr, "Error completing post to peer %u: %s\n",
              idx, ep->error());
      return ret;
    }
  }
  while (rc == -FI_EAGAIN);

  if (repostCnt > _stats._repostMax)
    _stats._repostMax = repostCnt;
  _stats._repostCnt += repostCnt;

  return 0;
}

#if 0                                   // No longer used
int EbLfBase::post(LocalIOVec& lclIov,
                   size_t      len,
                   unsigned    dst,
                   uint64_t    offset,
                   void*       ctx)
{
  //static unsigned wrtCnt  = 0;
  //static unsigned wrtCnt2 = 0;
  //static unsigned waitCnt = 0;

  //const struct iovec* iov = lclIov.iovecs();
  //void**              dsc = lclIov.desc();
  //
  //for (unsigned i = 0; i < lclIov.count(); ++i)
  //{
  //  printf("lclIov[%d]: base   = %p, size = %zd, desc = %p\n", i, iov[i].iov_base, iov[i].iov_len, dsc[i]);
  //}

  unsigned idx = _mappedId[dst];

  RemoteAddress rmtAdx(_ra[idx].rkey, _ra[idx].addr + offset, len);
  RemoteIOVec   rmtIov(&rmtAdx, 1);
  RmaMessage    rmaMsg(&lclIov, &rmtIov, ctx, offset); // imm_data is only 32 bits for verbs!

  //printf("rmtIov: rmtAdx = %p, size = %zd\n", (void*)rmtAdx.addr, len);

  //++wrtCnt;
  Endpoint* ep = _ep[idx];
  do
  {
    //++wrtCnt2;
    if (ep->writemsg(&rmaMsg, FI_REMOTE_CQ_DATA))  break;

    if (ep->error_num() == -FI_EAGAIN)
    {
      int              compCnt;
      fi_cq_data_entry cqEntry;
      const int        maxCnt = 1;

      ep->recv_comp_data();

      //printf("EbLfBase::post: Waiting for comp... %d of %d, %d\n", ++waitCnt, wrtCnt, wrtCnt2);
      if (!ep->comp_wait(&cqEntry, &compCnt, maxCnt))
      {
        if (ep->error_num() != -FI_EAGAIN)
        {
          fprintf(stderr, "Error completing operation with peer %u: %s\n",
                  idx, ep->error());
          return ep->error_num();
        }
      }
    }
    else
    {
      fprintf(stderr, "writemsg failed: %s\n", ep->error());
      return ep->error_num();
    }
  }
  while (ep->state() == EP_CONNECTED);

  return 0;
}
#endif
