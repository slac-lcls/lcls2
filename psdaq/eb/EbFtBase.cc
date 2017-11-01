#include "EbFtBase.hh"

#include "Endpoint.hh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace Pds;
using namespace Pds::Fabrics;
using namespace Pds::Eb;


EbFtBase::EbFtBase(unsigned nClients) :
  _ep(nClients),
  _mr(nClients),
  _ra(nClients),
  _cqPoller(NULL),
  _iSrc(0)
{
}

EbFtBase::~EbFtBase()
{
  unsigned nMr = _mr.size();
  unsigned nEp = _ep.size();

  for (unsigned i = 0; i < nEp; ++i)
    if (_cqPoller && _ep[i])  _cqPoller->del(_ep[i]);
  if (_cqPoller)  delete _cqPoller;

  for (unsigned i = 0; i < nMr; ++i)
    if (_mr[i])   delete _mr[i];
  for (unsigned i = 0; i < nEp; ++i)
    if (_ep[i])   delete _ep[i];
}

MemoryRegion* EbFtBase::registerMemory(void* buffer, size_t size)
{
  if (!_ep[0])  return NULL;

  Fabric* fab = _ep[0]->fabric();

  return fab->register_memory(buffer, size);
}

int EbFtBase::_syncLclMr(char*          region,
                         size_t         size,
                         Endpoint*      ep,
                         MemoryRegion*& mr)
{
  RemoteAddress ra(mr->rkey(), (uint64_t)region, size);
  memcpy(region, &ra, sizeof(ra));

  if (!ep->send_sync(region, sizeof(ra), mr))
  {
    fprintf(stderr, "Failed sending local memory specs to peer: %s\n",
            ep->error());
    return ep->error_num();
  }

  printf("Local  memory region: %p, size %zd\n", (void*)ra.addr, ra.extent);

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

  printf("Remote memory region: %p, size %zd\n", (void*)ra.addr, ra.extent);

  return 0;
}

uint64_t EbFtBase::_tryCq()
{
  // Cycle through all sources to find which one has data
  for (unsigned i = 0; i < _ep.size(); ++i)
  {
    if (!_ep[_iSrc])  continue;

    int              cqNum;
    fi_cq_data_entry cqEntry;

    if (_ep[_iSrc]->comp(&cqEntry, &cqNum, 1))
    {
      if (cqNum && (cqEntry.flags & (FI_REMOTE_WRITE | FI_REMOTE_CQ_DATA)))
      {
        // Revisit: Immediate data identifies which batch was written
        //          Better to use its address or parameters?
        //unsigned slot = (cqEntry.data >> 16) & 0xffff;
        //unsigned idx  =  cqEntry.data        & 0xffff;
        //batch = (Dgram*)&_pool[(slot * _maxBatches + idx) * _maxBatchSize];
        return cqEntry.data;
      }
    }
    else
    {
      if (_ep[_iSrc]->error_num() != -FI_EAGAIN)
      {
        fprintf(stderr, "Error completing operation with peer %u: %s\n",
                _iSrc, _ep[_iSrc]->error());
      }
    }
    if (++_iSrc == _ep.size())  _iSrc = 0;
  }

  return 0;
}

uint64_t EbFtBase::pend()
{
  uint64_t data = _tryCq();

  if (data)  return data;

  if (_cqPoller->poll())
  {
    data = _tryCq();
  }
  else
  {
    fprintf(stderr, "Error polling completion queues: %s",
            _cqPoller->error());
    return 0;
  }

  return data;
}

int EbFtBase::post(LocalIOVec& lclIov,
                   size_t      len,
                   unsigned    dst,
                   uint64_t    dstOffset,
                   void*       ctx)
{
  const struct iovec* iov = lclIov.iovecs();
  void**              dsc = lclIov.desc();

  for (unsigned i = 0; i < lclIov.count(); ++i)
  {
    printf("lclIov[%d]: base   = %p, size = %zd, desc = %p\n", i, iov[i].iov_base, iov[i].iov_len, dsc[i]);
  }

  RemoteAddress rmtAdx(_ra[dst].rkey, _ra[dst].addr + dstOffset, len);
  RemoteIOVec   rmtIov(&rmtAdx, 1);
  RmaMessage    rmaMsg(&lclIov, &rmtIov, ctx, rmtAdx.addr);

  printf("rmtIov[0]: rmtAdx = %p, size = %zd\n", (void*)rmtAdx.addr, len);

  while (1)
  {
    if (_ep[dst]->writemsg(&rmaMsg, FI_REMOTE_CQ_DATA))  break;

    if (_ep[dst]->error_num() == -FI_EAGAIN)
    {
      int              cqNum;
      fi_cq_data_entry cqEntry;

      printf("Waiting...\n");
      _ep[dst]->comp_wait(&cqEntry, &cqNum, 1);
      printf("Completion flags = 0x%016lx\n", cqEntry.flags);
    }
    else
    {
      fprintf(stderr, "writemsg failed: %s\n", _ep[dst]->error());
      return _ep[dst]->error_num();
    }
  }

  return 0;
}
