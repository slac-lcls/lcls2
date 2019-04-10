#include "EbLfLink.hh"

#include "Endpoint.hh"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

using namespace Pds;
using namespace Pds::Fabrics;
using namespace Pds::Eb;


EbLfLink::EbLfLink(Endpoint* ep,
                   unsigned  verbose,
                   uint64_t& pending) :
  _ep(ep),
  _mr(nullptr),
  _ra(),
  _depth(0),
  _count(1),
  _pending(pending),
  _id(-1),
  _verbose(verbose),
  _region(new char[sizeof(RemoteAddress)])
{
}

EbLfLink::EbLfLink(Endpoint* ep,
                   int       depth,
                   unsigned  verbose,
                   uint64_t& pending) :
  _ep(ep),
  _mr(nullptr),
  _ra(),
  _depth(depth),
  _count(1),
  _pending(pending),
  _id(-1),
  _verbose(verbose),
  _region(new char[sizeof(RemoteAddress)])
{
}

EbLfLink::~EbLfLink()
{
  if (_region)  delete [] _region;
}

int EbLfLink::preparePender(unsigned id)
{
  int           rc;
  uint32_t      sz;
  char          buf[sizeof(RemoteAddress)];
  MemoryRegion* mr;

  if ( (rc = setupMr(buf, sizeof(buf), &mr)) )   return rc;

  if ( (rc = recvU32(mr, &_id, "ID")) )          return rc;
  if ( (rc = sendU32(mr,   id, "ID")) )          return rc;

  if ( (rc = recvU32(mr, &sz, "region size")) )  return rc;
  if (sz != sizeof(RemoteAddress))               return -1;

  if ( (rc = sendMr(mr)) )                       return rc;

  return 0;
}

int EbLfLink::preparePender(unsigned id,
                            size_t*  size)
{
  int           rc;
  char          buf[sizeof(RemoteAddress)];
  MemoryRegion* mr;

  if ( (rc = setupMr(buf, sizeof(buf), &mr)) )   return rc;

  if ( (rc = recvU32(mr, &_id, "ID")) )          return rc;
  if ( (rc = sendU32(mr,   id, "ID")) )          return rc;

  uint32_t rs;
  if ( (rc = recvU32(mr, &rs, "region size")) )  return rc;
  *size = rs;

  // This method requires a call to setupMr(region, size) below
  // to complete the protocol, which involves a call to sendMr()

  return 0;
}

// A small memory region is needed in order to use the post(buf, len, immData)
// method, below.
int EbLfLink::preparePoster(unsigned id)
{
  return preparePoster(id, nullptr, 0, sizeof(RemoteAddress));
}

int EbLfLink::preparePoster(unsigned id,
                            void*    region,
                            size_t   size)
{
  return preparePoster(id, region, size, size);
}

// Buffers to be posted using the post(buf, len, offset, immData, ctx) method,
// below, must be covered by a memory region set up using this method.
int EbLfLink::preparePoster(unsigned id,
                            void*    region,
                            size_t   lclSize,
                            size_t   rmtSize)
{
  int    rc;
  size_t sz = sizeof(RemoteAddress);
  if (!_region)  return ENOMEM;

  if ( (rc = setupMr(_region, sz, &_mr)) )           return rc;

  if ( (rc = sendU32(_mr,   id, "ID")) )             return rc;
  if ( (rc = recvU32(_mr, &_id, "ID")) )             return rc;

  if ( (rc = sendU32(_mr, rmtSize, "region size")) ) return rc;
  if ( (rc = recvMr(_mr)) )                          return rc;

  // Region may already have stuff in it, so can't write on it above
  // Revisit: Would like to make it const, but has issues for Endpoint
  if (region)
  {
    if ( (rc = setupMr(region, lclSize, &_mr)) )     return rc;
  }

  return 0;
}

int EbLfLink::setupMr(void*  region,
                      size_t size)
{
  int rc;

  if ( (rc = setupMr(region, size, &_mr)) )  return rc;
  if ( (rc = sendMr(_mr)) )                  return rc;

  return rc;
}

int EbLfLink::setupMr(void*          region,
                      size_t         size,
                      MemoryRegion** mr)
{
  Fabric* fab = _ep->fabric();

  *mr = fab->register_memory(region, size);
  if (!*mr)
  {
    fprintf(stderr, "%s:\n  Failed to register memory region @ %p, size %zu: %s\n",
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

int EbLfLink::recvU32(MemoryRegion* mr,
                      uint32_t*     u32,
                      const char*   name)
{
  ssize_t rc;
  void*   buf = mr->start();

  if ((rc = _ep->recv_sync(buf, sizeof(u32), mr)) < 0)
  {
    fprintf(stderr, "%s:\n  Failed to receive %s from peer: %s\n",
            __PRETTY_FUNCTION__, name, _ep->error());
    return rc;
  }
  *u32 = *(uint32_t*)buf;

  if (_verbose)  printf("Received peer's %s: %d\n", name, *u32);

  return 0;
}

int EbLfLink::sendU32(MemoryRegion* mr,
                      uint32_t      u32,
                      const char*   name)
{
  ssize_t      rc;
  void*        buf = mr->start();

  LocalAddress adx(buf, sizeof(u32), mr);
  LocalIOVec   iov(&adx, 1);
  Message      msg(&iov, 0, NULL, 0);

  *(uint32_t*)buf = u32;
  if ((rc = _ep->sendmsg_sync(&msg, FI_TRANSMIT_COMPLETE | FI_COMPLETION)) < 0)
  {
    fprintf(stderr, "%s:\n  Failed to send %s to peer: %s\n",
            __PRETTY_FUNCTION__, name, _ep->error());
    return rc;
  }

  if (_verbose)  printf("Sent     peer   %s  %d\n", name, u32);

  return 0;
}

int EbLfLink::sendMr(MemoryRegion* mr)
{
  void*        buf = mr->start();
  size_t       len = mr->length();

  LocalAddress adx(buf, sizeof(_ra), mr);
  LocalIOVec   iov(&adx, 1);
  Message      msg(&iov, 0, NULL, 0);
  ssize_t      rc;

  _ra = RemoteAddress(mr->rkey(), (uint64_t)buf, len);
  memcpy(buf, &_ra, sizeof(_ra));

  if ((rc = _ep->sendmsg_sync(&msg, FI_COMPLETION)) < 0)
  {
    fprintf(stderr, "%s:\n  Failed to send local memory specs to ID %d: %s\n",
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

int EbLfLink::recvMr(MemoryRegion* mr)
{
  ssize_t  rc;
  if ((rc = _ep->recv_sync(mr->start(), sizeof(_ra), mr)) < 0)
  {
    fprintf(stderr, "%s:\n  Failed to receive remote region specs from ID %d: %s\n",
            __PRETTY_FUNCTION__, _id, _ep->error());
    return rc;
  }

  memcpy(&_ra, mr->start(), sizeof(_ra));

  if (_verbose)
  {
    printf("Received remote memory region: %10p : %10p, size %zd\n",
           (void*)_ra.addr, (void*)(_ra.addr + _ra.extent), _ra.extent);
  }

  return 0;
}

int EbLfLink::_postCompRecv(int count, void* ctx)
{
  int i;

  for (i = 0; i < count; ++i)
  {
    ssize_t rc;
    if ((rc = _ep->recv_comp_data(ctx)) < 0)
    {
      if (rc != -FI_EAGAIN)
        fprintf(stderr, "%s:\n  Failed to post a CQ buffer: %s\n",
                __PRETTY_FUNCTION__, _ep->error());
      break;
    }
  }

  return i;
}

// This method requires that the buffers to be posted are covered by a memory
// region set up using the preparePoster(id, region, size) method above.
int EbLfLink::post(const void* buf,
                   size_t      len,
                   uint64_t    offset,
                   uint64_t    immData,
                   void*       ctx)
{
  RemoteAddress ra(_ra.rkey, _ra.addr + offset, len);
  timespec      t0( {0, 0} );
  ssize_t       rc;

  _pending |= 1 << _id;

  while (true)
  {
    rc = _ep->write_data(buf, len, &ra, ctx, immData, _mr);
    if (!rc)  break;

    if (rc != -FI_EAGAIN)
    {
      fprintf(stderr, "%s:\n  write_data to ID %d failed: %s\n",
              __PRETTY_FUNCTION__, _id, _ep->error());
      break;
    }

    // With FI_SELECTIVE_COMPLETION, fabtests seems to indicate there is no need
    // to check the Tx completion queue as nothing will ever appear in it
    //fi_cq_data_entry cqEntry;
    //rc = _ep->txcq()->comp(&cqEntry, 1);
    //if ((rc < 0) && (rc != -FI_EAGAIN)) // EAGAIN means no completions available
    //{
    //  fprintf(stderr, "%s:\n  Error reading Tx CQ: %s\n",
    //          __PRETTY_FUNCTION__, _ep->txcq()->error());
    //  break;
    //}

    // Timeouts nominally occur only during shutdown
    // Revisit: This seems like the wrong thing to do since it can't be
    // guaranteed to happen only during shutdown, so maybe poll a flag?
    if (t0.tv_sec)
    {
      timespec t1;
      rc = clock_gettime(CLOCK_MONOTONIC_COARSE, &t1);
      if (rc < 0)  perror("clock_gettime");

      if (t1.tv_sec - t0.tv_sec > 4)
      {
        rc = -FI_ETIMEDOUT;
        break;
      }
    }
    else
    {
      rc = clock_gettime(CLOCK_MONOTONIC_COARSE, &t0);
      if (rc < 0)  perror("clock_gettime");
    }
  }

  _pending &= ~(1 << _id);

  return rc;
}

// This method requires that at least a small memory region has been registered.
// This can be done using the preparePoster(id, size) method above.
int EbLfLink::post(const void* buf,
                   size_t      len,
                   uint64_t    immData)
{
  if (ssize_t rc = _ep->inject_data(buf, len, immData) < 0)
  {
    fprintf(stderr, "%s:\n  inject_data failed: %s\n",
            __PRETTY_FUNCTION__, _ep->error());
    return rc;
  }

  return 0;
}
