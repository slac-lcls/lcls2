#include "EbLfLink.hh"

#include "Endpoint.hh"

#include "psdaq/service/fast_monotonic_clock.hh"

#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

using namespace Pds;
using namespace Pds::Fabrics;
using namespace Pds::Eb;


EbLfLink::EbLfLink(Endpoint* ep,
                   unsigned  verbose) :
  _ep(ep),
  _mr(nullptr),
  _id(-1),
  _verbose(verbose)
{
  if (setupMr(_buffer, sizeof(_buffer), &_bufMr))  return;
}

int EbLfLink::setupMr(void*  region,
                      size_t size)
{
  int rc;

  if (_mr)
  {
    Fabric* fab = _ep->fabric();

    if (_verbose)
    {
      printf("Deregistering   memory region: %10p : %10p, size %zd\n",
             _mr->start(), (char*)(_mr->start()) + _mr->length(), _mr->length());
    }
    if (!fab->deregister_memory(_mr))
    {
      fprintf(stderr, "%s:\n  Failed to deregister memory region %p (%p, %zd)\n",
              __PRETTY_FUNCTION__, _mr, _mr->start(), _mr->length());
    }
  }

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
           (*mr)->start(), (char*)((*mr)->start()) + (*mr)->length(), (*mr)->length());
  }

  return 0;
}

int EbLfLink::recvU32(uint32_t*     u32,
                      const char*   name)
{
  ssize_t rc;
  void*   buf = _bufMr->start();

  if ((rc = _ep->recv_sync(buf, sizeof(u32), _bufMr)) < 0)
  {
    fprintf(stderr, "%s:\n  Failed to receive %s from peer: %s\n",
            __PRETTY_FUNCTION__, name, _ep->error());
    return rc;
  }
  *u32 = *(uint32_t*)buf;

  if (_verbose)  printf("Received peer's %s: %d\n", name, *u32);

  return 0;
}

int EbLfLink::sendU32(uint32_t      u32,
                      const char*   name)
{
  ssize_t      rc;
  void*        buf = _bufMr->start();

  LocalAddress adx(buf, sizeof(u32), _bufMr);
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
  RemoteAddress ra(mr->rkey(), (uint64_t)mr->start(), mr->length());
  LocalAddress  adx(_bufMr->start(), sizeof(ra), _bufMr);
  LocalIOVec    iov(&adx, 1);
  Message       msg(&iov, 0, NULL, 0);
  ssize_t       rc;

  memcpy(adx.buf(), &ra, adx.len());

  if ((rc = _ep->sendmsg_sync(&msg, FI_COMPLETION)) < 0)
  {
    fprintf(stderr, "%s:\n  Failed to send local memory specs to ID %d: %s\n",
            __PRETTY_FUNCTION__, _id, _ep->error());
    return rc;
  }

  if (_verbose)
  {
    printf("Sent     local  memory region: %10p : %10p, size %zd\n",
           (void*)ra.addr, (void*)(ra.addr + ra.extent), ra.extent);
  }

  return 0;
}

int EbLfLink::recvMr(RemoteAddress& ra)
{
  ssize_t  rc;
  if ((rc = _ep->recv_sync(_bufMr->start(), sizeof(ra), _bufMr)) < 0)
  {
    fprintf(stderr, "%s:\n  Failed to receive remote region specs from ID %d: %s\n",
            __PRETTY_FUNCTION__, _id, _ep->error());
    return rc;
  }

  memcpy(&ra, _bufMr->start(), sizeof(ra));

  if (_verbose)
  {
    printf("Received remote memory region: %10p : %10p, size %zd\n",
           (void*)ra.addr, (void*)(ra.addr + ra.extent), ra.extent);
  }

  return 0;
}

// ---

EbLfSvrLink::EbLfSvrLink(Endpoint* ep,
                         int       depth,
                         unsigned  verbose) :
  EbLfLink(ep, verbose),
  _depth(depth),
  _count(1)
{
}

int EbLfSvrLink::exchangeIds(unsigned id)
{
  int rc;

  if ( (rc = recvU32(&_id, "ID")) )  return rc;
  if ( (rc = sendU32(  id, "ID")) )  return rc;

  return 0;
}

int EbLfSvrLink::prepare()
{
  int      rc;
  uint32_t sz;

  if ( (rc = recvU32(&sz, "region size")) )  return rc;
  if (sz != sizeof(_buffer))                 return -1;

  return 0;
}

int EbLfSvrLink::prepare(size_t* size)
{
  int      rc;
  uint32_t rs;

  if ( (rc = recvU32(&rs, "region size")) )  return rc;
  if (size)  *size = rs;

  // This method requires a call to setupMr(region, size) below
  // to complete the protocol, which involves a call to sendMr()

  return 0;
}

int EbLfSvrLink::_postCompRecv(int count)
{
  int i;

  for (i = _count; i < count; ++i)
  {
    ssize_t rc;
    if ((rc = _ep->recv_comp_data((void*)uintptr_t(i))) < 0)
    {
      if (rc != -FI_EAGAIN)
        fprintf(stderr, "%s:\n  Failed to post a CQ buffer %d: %s\n",
                __PRETTY_FUNCTION__, i, _ep->error());
      break;
    }
  }

  return i;
}

// ---

EbLfCltLink::EbLfCltLink(Endpoint* ep,
                         size_t    injectSize,
                         unsigned  verbose,
                         uint64_t& pending) :
  EbLfLink(ep, verbose),
  _injectSize(injectSize),
  _pending(pending)
{
}

int EbLfCltLink::exchangeIds(unsigned id)
{
  int rc;

  if ( (rc = sendU32(  id, "ID")) )  return rc;
  if ( (rc = recvU32(&_id, "ID")) )  return rc;

  return 0;
}

// A small memory region is needed in order to use the post(buf, len, immData)
// method, below.
int EbLfCltLink::prepare()
{
  return prepare(nullptr, 0, sizeof(RemoteAddress));
}

int EbLfCltLink::prepare(void*  region,
                         size_t size)
{
  return prepare(region, size, size);
}

// Buffers to be posted using the post(buf, len, offset, immData, ctx) method,
// below, must be covered by a memory region set up using this method.
int EbLfCltLink::prepare(void*  region,
                         size_t lclSize,
                         size_t rmtSize)
{
  int rc;

  if ( (rc = sendU32(rmtSize, "region size")) )  return rc;

  // Region may already have stuff in it, so can't write on it above
  // Revisit: Would like to make it const, but has issues in Endpoint.cc
  if (region)
  {
    if (_mr)
    {
      Fabric* fab = _ep->fabric();

      if (_verbose)
      {
        printf("Deregistering   memory region: %10p : %10p, size %zd\n",
               _mr->start(), (char*)(_mr->start()) + _mr->length(), _mr->length());
      }
      if (!fab->deregister_memory(_mr))
      {
        fprintf(stderr, "%s:\n  Failed to deregister memory region %p (%p, %zd)\n",
                __PRETTY_FUNCTION__, _mr, _mr->start(), _mr->length());
      }
    }

    if ( (rc = recvMr(_ra)) )                     return rc;
    if ( (rc = setupMr(region, lclSize, &_mr)) )  return rc;
  }

  return 0;
}

// This method requires that the buffers to be posted are covered by a memory
// region set up using the prepare(region, size) method above.
int EbLfCltLink::post(const void* buf,
                      size_t      len,
                      uint64_t    offset,
                      uint64_t    immData,
                      void*       ctx)
{
  RemoteAddress                    ra(_ra.rkey, _ra.addr + offset, len);
  ssize_t                          rc;
  fast_monotonic_clock::time_point t0;
  bool                             first = true;

  _pending |= 1 << _id;

  while (true)
  {
    if (len > _injectSize)
    {
      rc = _ep->writedata(buf, len, &ra, ctx, immData, _mr);
    }
    else
    {
      rc = _ep->inject_writedata(buf, len, &ra, immData);
    }
    if (!rc)  break;

    if (rc != -FI_EAGAIN)
    {
      fprintf(stderr, "%s:\n  writedata to ID %d failed: %s\n",
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

    if (!first)
    {
      using     ms_t  = std::chrono::milliseconds;
      auto      t1    = fast_monotonic_clock::now();
      const int msTmo = 5000;

      if (std::chrono::duration_cast<ms_t>(t1 - t0).count() > msTmo)
      {
        rc = -FI_ETIMEDOUT;
        break;
      }
    }
    else
    {
      t0    = fast_monotonic_clock::now();
      first = false;
    }
  }

  _pending &= ~(1 << _id);

  return rc;
}

// This method requires that at least a small memory region has been registered.
// This can be done using the prepare(size) method above.
int EbLfCltLink::post(const void* buf,
                      size_t      len,
                      uint64_t    immData)
{
  if (ssize_t rc = _ep->injectdata(buf, len, immData) < 0)
  {
    fprintf(stderr, "%s:\n  injectdata failed: %s\n",
            __PRETTY_FUNCTION__, _ep->error());
    return rc;
  }

  return 0;
}
