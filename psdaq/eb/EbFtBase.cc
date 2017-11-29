#include "EbFtBase.hh"

#include "Endpoint.hh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace Pds;
using namespace Pds::Fabrics;
using namespace Pds::Eb;


EbFtStats::EbFtStats(unsigned nPeers) :
  _postCnt(0),
  _repostCnt(0),
  _repostMax(0),
  _postWtAgnCnt(0),
  _postWtAgnMax(0),
  _pendCnt(0),
  _rependCnt(0),
  _rependMax(0),
  _rmtWrCnt(nPeers),
  _compAgnCnt(nPeers),
  _compAgnMax(nPeers),
  _compNoneCnt(0)
{
  for (unsigned i = 0; i < nPeers; ++i)
  {
    _rmtWrCnt[i]    = 0;
    _compAgnCnt[i]  = 0;
    _compAgnMax[i]  = 0;
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
  _postWtAgnCnt = 0;
  _postWtAgnMax = 0;
  _pendCnt      = 0;
  _rependCnt    = 0;
  _rependMax    = 0;
  _compNoneCnt  = 0;

  for (unsigned i = 0; i < _rmtWrCnt.size(); ++i)
  {
    _rmtWrCnt[i]    = 0;
    _compAgnCnt[i]  = 0;
    _compAgnMax[i]  = 0;
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

void EbFtStats::dump() const
{
  if (_postCnt)
  {
    printf("post: count %8ld, retries %8ld (max %8ld), waits %8ld (max %8ld)\n",
           _postCnt, _repostCnt, _repostMax, _postWtAgnCnt, _postWtAgnMax);
  }
  if (_pendCnt)
  {
    printf("pend: count %8ld, retries %8ld (max %8ld), None %8ld\n",
           _pendCnt, _rependCnt, _rependMax, _compNoneCnt);

    prtVec("rmtWrCnt",    _rmtWrCnt);
    prtVec("compAgnCnt",  _compAgnCnt);
    prtVec("compAgnMax",  _compAgnMax);
  }
}


EbFtBase::EbFtBase(unsigned nPeers) :
  _ep(nPeers),
  _lMr(nPeers),
  _rMr(nPeers),
  _ra(nPeers),
  _cqPoller(NULL),
  _id(nPeers),
  _mappedId(NULL),
  _stats(nPeers),
  _iSrc(0)
{
}

EbFtBase::~EbFtBase()
{
  if (_mappedId)  delete [] _mappedId;
}

void EbFtBase::registerMemory(void* buffer, size_t size)
{
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

uint64_t EbFtBase::_tryCq()
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

    if (ep->comp(&cqEntry, &compCnt, 1) && (compCnt == 1))
    {
      const unsigned flags = FI_REMOTE_WRITE | FI_REMOTE_CQ_DATA;

      ep->recv_comp_data();

      if ((cqEntry.flags & flags) == flags)
      {
        ++_stats._rmtWrCnt[iSrc];

        return _ra[iSrc].addr + cqEntry.data; // imm_data is only 32 bits for verbs!
      }

      fprintf(stderr, "Unexpected completion result with peer %u: count %d, flags %016lx\n",
              _id[iSrc], compCnt, cqEntry.flags);
      return 0;
    }
    ep->recv_comp_data();

    if (ep->error_num() != -FI_EAGAIN)
    {
      fprintf(stderr, "Error completing operation with peer %u: %s\n",
              _id[iSrc], ep->error());
      return 0;
    }

    ++_stats._compAgnCnt[iSrc];
  }

  ++_stats._compNoneCnt;

  return 0;
}

uint64_t EbFtBase::pend()
{
  uint64_t data;
  uint64_t rependCnt = 0;
  ++_stats._pendCnt;

  while ((data = _tryCq()) == 0)
  {
    if (!_cqPoller->poll())
    {
      fprintf(stderr, "Error polling completion queues: %s\n",
              _cqPoller->error());
      return 0;
    }
    ++rependCnt;
  }

  if (rependCnt > _stats._rependMax)
    _stats._rependMax = rependCnt;
  _stats._rependCnt += rependCnt;

  return data;
}

uint64_t EbFtBase::rmtAdx(unsigned dst, uint64_t offset)
{
  return _ra[_mappedId[dst]].addr + offset;
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

      //printf("EbFtBase::post: Waiting for comp... %d of %d, %d\n", ++waitCnt, wrtCnt, wrtCnt2);
      if (!ep->comp_wait(&cqEntry, &compCnt, 1))
      {
        if (ep->error_num() != -FI_EAGAIN)
        {
          fprintf(stderr, "Error completing operation with peer %u: %s\n",
                  idx, ep->error());
        }
      }
      ep->recv_comp_data();
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

int EbFtBase::post(const void* buf,
                   size_t      len,
                   unsigned    dst,
                   uint64_t    offset)
{
  void*    ctx = NULL;

  unsigned idx = _mappedId[dst];

  RemoteAddress rmtAdx(_ra[idx].rkey, _ra[idx].addr + offset, len);

  uint64_t repostCnt = 0;
  ++_stats._postCnt;

  Endpoint*     ep = _ep[idx];
  MemoryRegion* mr = _lMr[idx];
  while (!ep->write_data(const_cast<void*>(buf), len, &rmtAdx, ctx, offset, mr))
  {
    if (ep->state() != EP_CONNECTED)  return -1;

    if (ep->error_num() == -FI_EAGAIN)
    {
      int              compCnt;
      fi_cq_data_entry cqEntry;
      uint64_t         postWtAgnCnt = 0;

      while (!ep->comp_wait(&cqEntry, &compCnt, 1))
      {
        ep->recv_comp_data();

        if (ep->state()     != EP_CONNECTED)  return -1;
        if (ep->error_num() != -FI_EAGAIN)
        {
          fprintf(stderr, "Error completing operation with peer %u: %s\n",
                  idx, ep->error());
          return ep->error_num();
        }
        ++postWtAgnCnt;
      }
      ep->recv_comp_data();

      if (postWtAgnCnt > _stats._postWtAgnMax)
        _stats._postWtAgnMax = postWtAgnCnt;
      _stats._postWtAgnCnt += postWtAgnCnt;
    }
    else
    {
      fprintf(stderr, "writemsg failed: %s\n", ep->error());
      return ep->error_num();
    }

    ++repostCnt;
  }

  if (repostCnt > _stats._repostMax)
    _stats._repostMax = repostCnt;
  _stats._repostCnt += repostCnt;

  return 0;
}
