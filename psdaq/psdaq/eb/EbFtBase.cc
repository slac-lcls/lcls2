#include "EbFtBase.hh"

#include "Endpoint.hh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

using namespace Pds;
using namespace Pds::Fabrics;
using namespace Pds::Eb;


EbFtStats::EbFtStats(unsigned nPeers) :
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

EbFtStats::~EbFtStats()
{
}

void EbFtStats::clear()
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

void EbFtStats::dump()
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


EbFtBase::EbFtBase(unsigned nPeers) :
  _ep(nPeers),
  _lMr(nPeers),
  _rMr(nPeers),
  _ra(nPeers),
  _cqPoller(nullptr),
  //_rxDepth(nPeers),
  //_rOuts(nPeers),
  _id(nPeers),
  _mappedId(nullptr),
  _stats(nPeers),
  _iSrc(0)
{
}

EbFtBase::~EbFtBase()
{
  if (_mappedId)  delete [] _mappedId;
}

//#include <sys/mman.h>

void EbFtBase::registerMemory(void* buffer, size_t size)
{
  // No obvious effect:
  //int ret = mlock(buffer, size);
  //if (ret) perror ("mlock");

  for (unsigned i = 0; i < _ep.size(); ++i)
  {
    Endpoint* ep = _ep[i];

    if (!ep)  continue;

    Fabric* fab = ep->fabric();

    _lMr[i] = fab->register_memory(buffer, size);

    printf("Batch  memory region[%2d]: %p : %p, size %zd, desc %p\n",
           i, buffer, (char*)buffer + size, size, _lMr[i]->desc());
  }
}

void EbFtBase::_mapIds(unsigned nPeers)
{
  unsigned idMax = 0;
  for (unsigned i = 0; i < nPeers; ++i)
    if (_id[i] > idMax) idMax = _id[i];

  _mappedId = new unsigned[idMax + 1];
  assert(_mappedId);

  for (unsigned i = 0; i < nPeers; ++i)
    _mappedId[_id[i]] = i;
}

int EbFtBase::_syncLclMr(char*          region,
                         size_t         size,
                         Endpoint*      ep,
                         MemoryRegion*  mr,
                         RemoteAddress& ra)
{
  ra = RemoteAddress(mr->rkey(), (uint64_t)region, size);
  memcpy(region, &ra, sizeof(ra));

  if (!ep->send_sync(region, sizeof(ra), mr))
  {
    fprintf(stderr, "Failed sending local memory specs to peer: %s\n",
            ep->error());
    return ep->error_num();
  }

  printf("Local  memory region:     %p : %p, size %zd\n",
         (void*)ra.addr, (void*)(ra.addr + ra.extent), ra.extent);

  return 0;
}

int EbFtBase::_syncRmtMr(char*          region,
                         size_t         size,
                         Endpoint*      ep,
                         MemoryRegion*  mr,
                         RemoteAddress& ra)
{
  if (!ep->recv_sync(region, sizeof(ra), mr))
  {
    fprintf(stderr, "Failed receiving remote region specs from peer: %s\n",
            ep->error());
    perror("recv RemoteAddress");
    return ep->error_num();
  }

  memcpy(&ra, region, sizeof(ra));
  if (size > ra.extent)
  {
    fprintf(stderr, "Remote region size (%lu) is less than required (%lu)\n",
            ra.extent, size);
    return -1;
  }

  printf("Remote memory region:     %p : %p, size %zd\n",
         (void*)ra.addr, (void*)(ra.addr + ra.extent), ra.extent);

  return 0;
}

const EbFtStats& EbFtBase::stats() const
{
  return _stats;
}

uint64_t EbFtBase::rmtAdx(unsigned dst, uint64_t offset) const
{
  return _ra[_mappedId[dst]].addr + offset;
}

int EbFtBase::_postCompRecv(Endpoint* ep, unsigned count)
{
  unsigned i;

  for (i = 0; i < count; ++i)
  {
    if (!ep->recv_comp_data())
    {
      fprintf(stderr, "Failed to post a CQ buffer: %s\n", ep->error());
      break;
    }
  }

  return i;
}

int EbFtBase::_tryCq(uint64_t* data)
{
  // Cycle through all sources to find which one has data
  for (unsigned i = 0; i < _ep.size(); ++i)
  {
    unsigned iSrc = _iSrc++;
    if (_iSrc == _ep.size())  _iSrc = 0;

    Endpoint* ep = _ep[iSrc];
    if (!ep)  continue;

    int              compCnt;
    fi_cq_data_entry cqEntry;
    const int        maxCnt = 1;
    bool             ret    = ep->comp(&cqEntry, &compCnt, maxCnt);
    if (ret && (compCnt == maxCnt))
    {
      const unsigned flags = FI_REMOTE_WRITE | FI_REMOTE_CQ_DATA;

      if ((cqEntry.flags & flags) == flags)
      {
        ++_stats._rmtWrCnt[iSrc];

        //if (--_rOuts[iSrc] <= 1)
        //{
        //  _rOuts[iSrc] += _postCompRecv(ep, _rxDepth[iSrc] - _rOuts[iSrc]);
        //  if (_rOuts[iSrc] < _rxDepth[iSrc])
        //  {
        //    fprintf(stderr, "Pend: Posted only %d of %d CQ buffers at index %d(%d)\n",
        //            _rOuts[iSrc], _rxDepth[iSrc], i, _id[iSrc]);
        //  }
        //}
        if (!ep->recv_comp_data())
        {
          if (ep->error_num() != -FI_EAGAIN)
            fprintf(stderr, "Pend: recv_comp_data() error: %s\n", ep->error());
        }

        *data = _ra[iSrc].addr + cqEntry.data; // imm_data is only 32 bits for verbs!
        return 0;
      }

      fprintf(stderr, "Unexpected completion queue entry of peer %u: count %d, flags %016lx\n",
              _id[iSrc], compCnt, cqEntry.flags);
      return -1;
    }

    if (ep->error_num() != -FI_EAGAIN)
    {
      fprintf(stderr, "Error reading completion queue of peer %u: %s\n",
              _id[iSrc], ep->error());
      return ep->error_num();
    }

    ++_stats._compAgnCnt[iSrc];
  }

  return -FI_EAGAIN;
}

int EbFtBase::pend(uint64_t* data)
{
  int      ret;
  uint64_t rependCnt  = 0;
  ++_stats._pendCnt;

  while ((ret = _tryCq(data)) == -FI_EAGAIN)
  {
    const int tmo = 5000;               // milliseconds
    if (!_cqPoller->poll(tmo))
    {
      if (_cqPoller->error_num() == -FI_ETIMEDOUT)
      {
        ++_stats._pendTmoCnt;
      }
      else
      {
        fprintf(stderr, "Error polling completion queues: %s\n",
                _cqPoller->error());
      }
      return _cqPoller->error_num();
    }
    ++rependCnt;
  }

  if (rependCnt > _stats._rependMax)
    _stats._rependMax = rependCnt;
  _stats._rependCnt += rependCnt;

  return ret;
}

int EbFtBase::post(const void*    buf,
                   size_t         len,
                   unsigned       dst,
                   uint64_t       offset)
{
  uint64_t repostCnt = 0;
  ++_stats._postCnt;

  unsigned      idx = _mappedId[dst];
  Endpoint*     ep  = _ep[idx];
  RemoteAddress ra   (_ra[idx].rkey, _ra[idx].addr + offset, len);
  MemoryRegion* mr  = _lMr[idx];
  void*         ctx = nullptr;
  int           rc  = 0;
  do
  {
    if (!ep->write_data(buf, len, &ra, ctx, offset, mr))
    {
      if ((rc = ep->error_num()) != -FI_EAGAIN)
      {
        fprintf(stderr, "write_data failed: %s\n", ep->error());
        --repostCnt;
      }
      ++repostCnt;
    }

    // It appears that the CQ must always be read, not just after EAGAIN
    int              compCnt;
    fi_cq_data_entry cqEntry;
    const ssize_t    maxCnt = 1;
    const int        tmo    = 5000;   // milliseconds
    if (!ep->comp_wait(&cqEntry, &compCnt, maxCnt, tmo))
    {
      if (ep->error_num() != -FI_EAGAIN) // EAGAIN => timeout or signal
      {                                  // Occurs when peer disconnects
        fprintf(stderr, "Error completing post to peer %u: %s\n",
                idx, ep->error());
      }
      rc = ep->error_num();
      //ep->recv_comp_data();
      return rc;
    }

    //if (--_rOuts[idx] <= 1)
    //{
    //  _rOuts[idx] += _postCompRecv(ep, _rxDepth[idx] - _rOuts[idx]);
    //  if (_rOuts[idx] < _rxDepth[idx])
    //  {
    //    fprintf(stderr, "Post: Posted only %d of %d CQ buffers at index %d(%d)\n",
    //            _rOuts[idx], _rxDepth[idx], idx, dst);
    //  }
    //}
    //if (!ep->recv_comp_data())
    //{
    //  if (ep->error_num() != -FI_EAGAIN)
    //    fprintf(stderr, "Post: recv_comp_data() error: %s\n", ep->error());
    //}
  }
  while (rc == -FI_EAGAIN);

  if (repostCnt > _stats._repostMax)
    _stats._repostMax = repostCnt;
  _stats._repostCnt += repostCnt;

  return 0;
}

#if 0                                   // No longer used
int EbFtBase::post(LocalIOVec& lclIov,
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

      //printf("EbFtBase::post: Waiting for comp... %d of %d, %d\n", ++waitCnt, wrtCnt, wrtCnt2);
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
