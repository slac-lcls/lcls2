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
  _ra(nClients)
{
}

EbFtBase::~EbFtBase()
{
  unsigned nMr = _mr.size();
  unsigned nEp = _ep.size();

  for (unsigned i = 0; i < nMr; ++i)
    if (_mr[i])   delete _mr[i];
  for (unsigned i = 0; i < nEp; ++i)
    if (_ep[i])   delete _ep[i];
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

uint64_t EbFtBase::pend()
{
  while (1)
  {
    // Cycle through all sources and don't starve any one
    for (unsigned i = 0; i < _ep.size(); ++i) // Revisit: Awaiting select() or poll() soln
    {
      int              cqNum;
      fi_cq_data_entry cqEntry;

      if (_ep[i]->comp(&cqEntry, &cqNum, 1)) // Revisit: Wait on all EPs with tmo
      {
        if ((cqNum == 1) && (cqEntry.flags & FI_REMOTE_WRITE))
        {
          // Revisit: Immediate data identifies which batch was written
          //          Better to use its address or parameters?
          //unsigned slot = (cqEntry.data >> 16) & 0xffff;
          //unsigned idx  =  cqEntry.data        & 0xffff;
          //batch = (Dgram*)&_pool[(slot * _maxBatches + idx) * _maxBatchSize];
          return cqEntry.data;
        }
      }
    }
  }
}

int EbFtBase::post(fi_msg_rma* msg,
                   unsigned    dst,
                   uint64_t    dstOffset,
                   void*       ctx)
{
  fi_rma_iov* iov = const_cast<fi_rma_iov*>(msg->rma_iov);
  iov->addr = _ra[dst].addr + dstOffset;
  iov->key  = _ra[dst].rkey;             // Revisit: _mr's?

  void* desc = _mr[dst]->desc();         // Revisit: Fix kluginess
  for (unsigned i = 0; i < msg->iov_count; ++i)
  {
    msg->desc[i] = desc;
  }

  msg->data    = iov->addr;
  msg->context = ctx;

  if (!_ep[dst]->write_msg(msg, FI_REMOTE_CQ_DATA)) // Revisit: Flags?
  {
    int errNum = _ep[dst]->error_num();
    fprintf(stderr, "%s() failed, ret = %d (%s)\n",
            "write_msg", errNum, _ep[dst]->error());
    return errNum;
  }

  //_ep[dst]->writeMsg(batch->iov(), batch->iovCount(), dstAddr, immData,
  //                   _ra[dst], _mr[dst]);

  return 0;
}
