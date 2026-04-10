#include "EbLfLink.hh"

#include "Endpoint.hh"

#include "psdaq/service/fast_monotonic_clock.hh"

#include <chrono>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

using namespace Pds;
using namespace Pds::Fabrics;
using namespace Pds::Eb;

using ms_t = std::chrono::milliseconds;


static int checkMr(Fabric*         fabric,
                   void*           region,
                   size_t          size,
                   MemoryRegion*   mr,
                   const unsigned& verbose)
{
  // If the region (with any size) is already registered as mr...
  if (mr == fabric->lookup_memory(region, sizeof(uint8_t)))
  {
    // and if its size fits, mr can be reused
    if ((region == mr->start()) && (size <= mr->length()))
    {
      if (verbose)
      {
        fprintf(stderr, "Reusing        MR: %10p : %10p, size 0x%08zx = %zu\n",
                mr->start(), (char*)(mr->start()) + mr->length(),
                mr->length(), mr->length());
      }
      return 0;
    }
  }
  // If region has been assigned to a different MR, or it doesn't fit,
  // deregister mr so that it can be updated
  if (!fabric->deregister_memory(mr))
  {
    fprintf(stderr, "%s:\n  Failed to deregister MR %p (%p, %zu)\n",
            __PRETTY_FUNCTION__, mr, mr->start(), mr->length());
  }
  if (verbose > 1)
  {
    fprintf(stderr, "Freed          MR: %10p : %10p, size 0x%08zx = %zu\n",
            mr->start(), (char*)(mr->start()) + mr->length(),
            mr->length(), mr->length());
  }
  return 1;
}

int Pds::Eb::setupMr(Fabric*         fabric,
                     void*           region,
                     size_t          size,
                     MemoryRegion**  memReg,
                     const unsigned& verbose)
{
  // If *memReg describes a region, check that its size is appropriate
  if (*memReg && !checkMr(fabric, region, size, *memReg, verbose))
  {
    return 0;
  }

  auto t0(std::chrono::steady_clock::now());
  auto mr = fabric->register_memory(region, size);
  auto t1 = std::chrono::steady_clock::now();
  auto dT = std::chrono::duration_cast<ms_t>(t1 - t0).count();
  *memReg = mr;                         // Even on error, set *memReg
  if (!mr)
  {
    fprintf(stderr, "%s:\n  Failed to register MR @ %p, size %zu  (%lu ms): %s\n",
            __PRETTY_FUNCTION__, region, size, dT, fabric->error());
    return fabric->error_num();
  }

  if (verbose)
  {
    fprintf(stderr, "Registered     MR: %10p : %10p, size 0x%08zx = %zu  (%lu ms)\n",
            mr->start(), (char*)(mr->start()) + mr->length(),
            mr->length(), mr->length(), dT);
  }

  return 0;
}

// ---

EbLfLink::EbLfLink(Endpoint*          ep,
                   const unsigned     depth,
                   const unsigned&    verbose,
                   volatile uint64_t& pending,
                   volatile uint64_t& posting) :
  _id      (-1),
  _ep      (ep),
  _mr      (nullptr),
  _verbose (verbose),
  _timedOut(0ull),
  _pending (pending),
  _posting (posting),
  _depth   (depth),
  _credits (0)
{
  _pending = 0;
  _posting = 0;

  postCompRecv(0);
}

EbLfLink::~EbLfLink()
{
  _pending = 0;
  _posting = 0;

  if (_credits != _depth)
    fprintf(stderr, "%s:\n  *** link %p, id %d, _credits (%u) != _depth (%u)\n",
            __PRETTY_FUNCTION__, this, _id, _credits, _depth);
}

int EbLfLink::recvU32(uint32_t*   u32,
                      const char* peer,
                      const char* name)
{
  ssize_t  rc;
  uint64_t data;
  if ((rc = poll(&data, 7000)))
  {
    const char* errMsg = rc == -FI_EAGAIN ? "Timed out" : _ep->error();
    fprintf(stderr, "%s:\n  Failed to receive %s from %s: %s\n",
            __PRETTY_FUNCTION__, name, peer, errMsg);
    return rc;
  }
  *u32 = data;

  if (_verbose > 1)  fprintf(stderr, "Received %s's %s: 0x%08x = %d\n",
                             peer, name, *u32, *u32);

  return 0;
}

int EbLfLink::sendU32(uint32_t    u32,
                      const char* peer,
                      const char* name)
{
  ssize_t  rc;
  uint64_t imm = u32;
  if ((rc = post(imm)))
  {
    const char* errMsg = rc == -FI_EAGAIN ? "Timed out" : _ep->error();
    fprintf(stderr, "%s:\n  Failed to send %s to %s: %s\n",
            __PRETTY_FUNCTION__, name, peer, errMsg);
    return rc;
  }

  if (_verbose > 1)  fprintf(stderr, "Sent     %s   %s  0x%08x = %d\n",
                             peer, name, u32, u32);

  return 0;
}

int EbLfLink::recvMr(RemoteAddress& ra,
                     const char*    peer)
{
  ssize_t   rc;
  unsigned* ptr = reinterpret_cast<unsigned*>(&ra);

  for (unsigned i = 0; i < sizeof(ra)/sizeof(*ptr); ++i)
  {
    uint64_t imm;
    if ((rc = poll(&imm, 7000)))
    {
      const char* errMsg = rc == -FI_EAGAIN ? "Timed out" : _ep->error();
      fprintf(stderr, "%s:\n  Failed to receive %s from %s ID %d: %s\n",
              __PRETTY_FUNCTION__, "remote region specs", peer, _id, errMsg);
      return rc;
    }
    *ptr++ = imm & 0x00000000ffffffffull;
  }

  if (_verbose > 1)
  {
    fprintf(stderr, "Received %s's MR: %10p : %10p, size 0x%08zx = %zu\n", peer,
            (void*)ra.addr, (void*)(ra.addr + ra.extent), ra.extent, ra.extent);
  }

  return 0;
}

int EbLfLink::sendMr(MemoryRegion* mr,
                     const char*   peer)
{
  ssize_t       rc;
  RemoteAddress ra(mr->rkey(), (uint64_t)mr->start(), mr->length());
  unsigned*     ptr = reinterpret_cast<unsigned*>(&ra);

  for (unsigned i = 0; i < sizeof(ra)/sizeof(*ptr); ++i)
  {
    uint64_t imm = *ptr++;
    if ((rc = post(imm)) < 0)
    {
      const char* errMsg = rc == -FI_EAGAIN ? "Timed out" : _ep->error();
      fprintf(stderr, "%s:\n  Failed to send %s to %s ID %d: %s\n",
              __PRETTY_FUNCTION__, "local memory specs", peer, _id, errMsg);
      return rc;
    }
  }

  if (_verbose > 1)
  {
    fprintf(stderr, "Sent     %s   MR: %10p : %10p, size 0x%08zx = %zu\n", peer,
            (void*)ra.addr, (void*)(ra.addr + ra.extent), ra.extent, ra.extent);
  }

  return 0;
}

// ---

EbLfSvrLink::EbLfSvrLink(Endpoint*          ep,
                         const unsigned     depth,
                         const unsigned&    verbose,
                         volatile uint64_t& pending,
                         volatile uint64_t& posting) :
  EbLfLink(ep, depth, verbose, pending, posting)
{
}

int EbLfSvrLink::_synchronizeBegin()
{
  int rc;

  time_t rawTime{time(NULL)};           // Current time
  fprintf(stderr, "%s:  Id %d, start @ %s", __PRETTY_FUNCTION__, _id, ctime(&rawTime));

  // Send a synchronization message to _one_ client
  uint64_t imm = _BegSync;              // Use a different value from Clients
  if ( (rc = EbLfLink::post(imm)) )  return rc;


  // Drain any stale transmissions that are stuck in the pipe
  auto t0{std::chrono::steady_clock::now()};
  while ((rc = EbLfLink::poll(&imm, 60000)) == 0)
  {
    if (imm == _EndSync)  break;        // Break on synchronization message

    if (_verbose)
      fprintf(stderr, "%s:  Got junk from id %d: imm %08lx != %08x\n",
              __PRETTY_FUNCTION__, _id, imm, _EndSync);
  }
  auto t1{std::chrono::steady_clock::now()};

  if (rc == -FI_EAGAIN)
    fprintf(stderr, "%s:  Timed out (%lu ms), id %d: imm %08lx != %08x\n",
            __PRETTY_FUNCTION__, std::chrono::duration_cast<ms_t>(t1-t0).count(),
            _id, imm, _EndSync);

  rawTime = time(NULL);                 // Current time
  fprintf(stderr, "%s:  Id %d, end   @ %s", __PRETTY_FUNCTION__, _id, ctime(&rawTime));

  return rc;
}

int EbLfSvrLink::_synchronizeEnd()
{
  int rc;

  uint64_t imm = _SvrSync;
  if ( (rc = EbLfLink::post(imm)) )  return rc;

  // Drain any stale transmissions that are stuck in the pipe
  while ((rc = EbLfLink::poll(&imm, 7000)) == 0)
  {
    if (imm == _CltSync)  break;

    if (_verbose)
      fprintf(stderr, "%s:  Got junk from id %d: imm %08lx != %08x\n",
              __PRETTY_FUNCTION__, _id, imm, _CltSync);
  }

  if (rc == -FI_EAGAIN)
    fprintf(stderr, "\n%s:  Timed out\n\n", __PRETTY_FUNCTION__);

  return rc;
}

int EbLfSvrLink::exchangeId(unsigned    id,
                            const char* peer)
{
  int rc;

  // Wait for synchronization to complete successfully prior to any sends/recvs
  if ( (rc = _synchronizeBegin()) )
  {
    fprintf(stderr, "%s:\n  Failed synchronize Begin with %s: rc %d\n",
            __PRETTY_FUNCTION__, peer, rc);
    return rc;
  }

  // Exchange IDs
  if ( (rc = recvU32(&_id, peer, "ID")) )  return rc;
  if ( (rc = sendU32(  id, peer, "ID")) )  return rc;

  // Verify the exchanges are complete
  if ( (rc = _synchronizeEnd()) )
  {
    fprintf(stderr, "%s:\n  Failed synchronize End with %s: rc %d\n",
            __PRETTY_FUNCTION__, peer, rc);
    return rc;
  }

  return 0;
}

int EbLfSvrLink::prepare(size_t*     size,
                         const char* peer)
{
  int      rc;
  uint32_t rs;

  // Wait for synchronization to complete successfully prior to any sends/recvs
  if ( (rc = _synchronizeBegin()) )
  {
    fprintf(stderr, "%s:\n  Failed synchronize Begin with %s: rc %d\n",
            __PRETTY_FUNCTION__, peer, rc);
    return rc;
  }

  // Get MR size
  if ( (rc = recvU32( &rs, peer, "sz")) )  return rc;
  if (size)  *size = rs;

  // This method requires a call to setupMr(region, size) below
  // to complete the protocol, which involves a call to sendMr()

  return 0;
}

int EbLfSvrLink::setupMr(void*       region,
                         size_t      size,
                         const char* peer)
{
  int rc;

  // Set up the MR and provide its specs to the other side
  if ( (rc = Pds::Eb::setupMr(_ep->fabric(), region, size, &_mr, _verbose)) )  return rc;
  if ( (rc = sendMr(_mr, peer)) )  return rc;

  // Verify the exchanges are complete
  if ( (rc = _synchronizeEnd()) )
  {
    fprintf(stderr, "%s:\n  Failed synchronize End with %s: rc %d\n",
            __PRETTY_FUNCTION__, peer, rc);
    return rc;
  }

  return rc;
}

// ---

EbLfCltLink::EbLfCltLink(Endpoint*          ep,
                         const unsigned     depth,
                         const unsigned&    verbose,
                         volatile uint64_t& pending,
                         volatile uint64_t& posting) :
  EbLfLink(ep, depth, verbose, pending, posting)
{
}

int EbLfCltLink::setupMr(void* region, size_t size)
{
  if (_ep)
    return Pds::Eb::setupMr(_ep->fabric(), region, size, &_mr, _verbose);

  return -1;
}

int EbLfCltLink::_synchronizeBegin()
{
  int rc;

  time_t rawTime{time(NULL)};           // Current time
  fprintf(stderr, "%s:  Id %d, start @ %s", __PRETTY_FUNCTION__, _id, ctime(&rawTime));

  // NB: Clients can't send anything to a server before receiving the
  //     synchronization message or the Server will get confused
  // Drain any stale transmissions that are stuck in the pipe
  uint64_t imm;
  auto t0{std::chrono::steady_clock::now()};
  while ((rc = EbLfLink::poll(&imm, 60000)) == 0)
  {
    if (imm == _BegSync)  break;        // Break on synchronization message

    if (_verbose)
      fprintf(stderr, "%s:  Got junk from id %d: imm %08lx != %08x\n",
              __PRETTY_FUNCTION__, _id, imm, _BegSync);
  }
  auto t1{std::chrono::steady_clock::now()};

  if (rc == -FI_EAGAIN)
    fprintf(stderr, "%s:  Timed out (%lu ms), id %d: imm %08lx != %08x\n",
            __PRETTY_FUNCTION__, std::chrono::duration_cast<ms_t>(t1-t0).count(),
            _id, imm, _BegSync);

  if (rc == 0)
  {
    // Send a synchronization message to the server
    imm = _EndSync;                       // Use a different value from Servers
    if ( (rc = EbLfLink::post(imm)) )  return rc;
  }

  rawTime = time(NULL);                 // Current time
  fprintf(stderr, "%s:  Id %d, end   @ %s", __PRETTY_FUNCTION__, _id, ctime(&rawTime));

  return rc;
}

int EbLfCltLink::_synchronizeEnd()
{
  int rc;

  uint64_t imm = _CltSync;
  if ( (rc = EbLfLink::post(imm)) )  return rc;

  // Drain any stale transmissions that are stuck in the pipe
  while ((rc = EbLfLink::poll(&imm, 7000)) == 0)
  {
    if (imm == _SvrSync)  break;

    if (_verbose)
      fprintf(stderr, "%s:  Got junk from id %d: imm %08lx != %08x\n",
              __PRETTY_FUNCTION__, _id, imm, _SvrSync);
  }

  if (rc == -FI_EAGAIN)
    fprintf(stderr, "\n%s:  Timed out\n\n", __PRETTY_FUNCTION__);

  return rc;
}

int EbLfCltLink::exchangeId(unsigned    id,
                            const char* peer)
{
  int rc;

  // Wait for synchronization to complete successfully prior to any sends/recvs
  if ( (rc = _synchronizeBegin()) )
  {
    fprintf(stderr, "%s:\n  Failed synchronize Begin with %s: rc %d\n",
            __PRETTY_FUNCTION__, peer, rc);
    return rc;
  }

  // Exchange IDs and get MR size
  if ( (rc = sendU32(  id, peer, "ID")) )  return rc;
  if ( (rc = recvU32(&_id, peer, "ID")) )  return rc;

  // Verify the exchanges are complete
  if ( (rc = _synchronizeEnd()) )
  {
    fprintf(stderr, "%s:\n  Failed synchronize End with peer: rc %d\n",
            __PRETTY_FUNCTION__, rc);
    return rc;
  }

  return 0;
}

int EbLfCltLink::prepare(void*       region,
                         size_t      size,
                         const char* peer)
{
  return prepare(region, size, size, peer);
}

// Buffers to be posted using the post(buf, len, offset, immData, ctx) method,
// below, must be covered by a memory region set up using this method.
int EbLfCltLink::prepare(void*       region,
                         size_t      lclSize,
                         size_t      rmtSize,
                         const char* peer)
{
  int rc;

  // Wait for synchronization to complete successfully prior to any sends/recvs
  if ( (rc = _synchronizeBegin()) )
  {
    fprintf(stderr, "%s:\n  Failed synchronize Begin with %s: rc %d\n",
            __PRETTY_FUNCTION__, peer, rc);
    return rc;
  }

  // Revisit: Would like to make it const, but has issues in Endpoint.cc
  if (region)
  {
    if ( (rc = sendU32(rmtSize, peer, "sz")) )  return rc;

    // Set up the MR and provide its specs to the other side
    if ( (rc = Pds::Eb::setupMr(_ep->fabric(), region, lclSize, &_mr, _verbose)) )  return rc;
    if ( (rc = recvMr (_ra, peer)) )  return rc;
  }

  // Verify the exchanges are complete
  if ( (rc = _synchronizeEnd()) )
  {
    fprintf(stderr, "%s:\n  Failed synchronize End with peer: rc %d\n",
            __PRETTY_FUNCTION__, rc);
    return rc;
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
  RemoteAddress ra{_ra.rkey, _ra.addr + offset, len};
  auto          t0{fast_monotonic_clock::now()};
  ssize_t       rc;

  _posting |= 1 << _id;

  while (true)
  {
    // writedata() will do (the equivalent of) inject_writedata() if the size
    // is less than the inject_size, so there is no need to duplicate that here
    if ( !(rc = _ep->writedata(buf, len, &ra, ctx, immData, _mr)) )  break;

    if (rc != -FI_EAGAIN)
    {
      fprintf(stderr, "%s:\n  writedata() to ID %d failed: %s\n",
              __PRETTY_FUNCTION__, _id, _ep->error());
      break;
    }

    const ms_t tmo{300000};            // Revisit: Skip the timeout altogether?
    auto       t1 {fast_monotonic_clock::now()};

    if (t1 - t0 > tmo)
    {
      ++_timedOut;
      fprintf(stderr, "%s:\n  writedata() to ID %d timed out (%lu ms): %s\n",
              __PRETTY_FUNCTION__, _id, tmo.count(), _ep->error());
      break;
    }
  }

  _posting &= ~(1 << _id);

  return rc;
}

// This method requires that a memory region has been registered that covers the
// buffer specified by buf, len.  This can be done using the prepare(size)
// method above.  If these are NULL, no memory region is needed.
int EbLfLink::post(const void* buf,
                   size_t      len,
                   uint64_t    immData)
{
  auto    t0{fast_monotonic_clock::now()};
  ssize_t rc;

  _posting |= 1 << _id;

  while (true)
  {
    if ( !(rc = _ep->injectdata(buf, len, immData)) )  break;

    if (rc != -FI_EAGAIN)
    {
      fprintf(stderr, "%s:\n  injectdata() to ID %d failed: %s\n",
              __PRETTY_FUNCTION__, _id, _ep->error());
      break;
    }

    const ms_t tmo{300000};            // Revisit: Skip the timeout altogether?
    auto       t1 {fast_monotonic_clock::now()};

    if (t1 - t0 > tmo)
    {
      ++_timedOut;
      fprintf(stderr, "%s:\n  injectdata() to ID %d timed out (%lu ms): %s\n",
              __PRETTY_FUNCTION__, _id, tmo.count(), _ep->error());
      break;
    }
  }

  _posting &= ~(1 << _id);

  return rc;
}

static bool isDisconnected(EventQueue* eq)
{
  bool                  cmEntry;
  struct fi_eq_cm_entry entry;
  uint32_t              event;

  // @todo: Should have a 3rd state: true, false and don't know (error)
  return (eq->event(&event, &entry, &cmEntry) &&
          cmEntry && (event == FI_SHUTDOWN));
}

static int dumpCqError(unsigned id, CompletionQueue* cq, const char* fn)
{
  struct fi_cq_err_entry compErr;
  auto rc = cq->comp_error(&compErr);
  if (rc > 0)
  {
    fprintf(stderr, "%s:\n  No CQ entries for ID %d:\n", fn, id);
    cq->comp_error_dump(compErr);
    rc = -compErr.err;                  // Return the real error
  }
  else
    fprintf(stderr, "%s:\n  Error decoding Rx CQ error: rc %zd: %s\n",
            fn, rc, cq->error());

  return rc;
}

int EbLfLink::poll(uint64_t* data)      // Sample only, don't wait
{
  int              rc;
  fi_cq_data_entry cqEntry;
  auto             cq = _ep->rxcq();

  rc = cq->comp(&cqEntry, 1);
  if (postCompRecv(rc > 0 ? rc : 0))
  {
    fprintf(stderr, "%s:\n  Failed to post %d CQ buffers\n",
            __PRETTY_FUNCTION__, rc);
  }

  if (rc > 0)
  {
    *data = cqEntry.data;

    return 0;
  }

  if (rc == -FI_EAGAIN)
    ++_timedOut;
  else if (isDisconnected(_ep->eq()))   // rc uninteresting if disconnected
  {
    fprintf(stderr, "%s:\n  Link ID %d is disconnected\n",
            __PRETTY_FUNCTION__, _id);
    rc = -FI_ENOTCONN;
  }
  else                                 // Notify of unexpected error
  {
    if (rc == -FI_EAVAIL)              // Return actual error instead of EAVAIL
      rc = dumpCqError(_id, cq, __PRETTY_FUNCTION__);
    else
      fprintf(stderr, "%s:\n  No CQ entries for ID %d: rc %d: %s\n",
              __PRETTY_FUNCTION__, _id, rc, cq->error());
  }
  return rc;
}

int EbLfLink::poll(uint64_t* data, int msTmo) // Wait until timed out
{
  int  rc;
  auto cq = _ep->rxcq();
  auto t0{fast_monotonic_clock::now()};

  class Pending
  {
  public:
    Pending(volatile uint64_t& pending) : _pending(pending) { _pending += 1; }
    ~Pending() { _pending -= 1; }
  private:
    volatile uint64_t& _pending;
  } pending(_pending);

  do
  {
    fi_cq_data_entry cqEntry;
    rc = cq->comp_wait(&cqEntry, 1, msTmo);
    if (postCompRecv(rc > 0 ? rc : 0))
    {
      fprintf(stderr, "%s:\n  Failed to post %d CQ buffers\n",
              __PRETTY_FUNCTION__, rc);
    }

    if (rc > 0)
    {
      *data = cqEntry.data;

      return 0;
    }
    if (rc == -FI_EAGAIN)
    {
      const ms_t tmo{msTmo};
      auto       t1 {fast_monotonic_clock::now()};
      if (t1 - t0 > tmo)
      {
        ++_timedOut;
        return rc;
      }
    }
  }
  while ((rc == -FI_EAGAIN) || (rc == 0));

  if (isDisconnected(_ep->eq()))        // rc uninteresting if disconnected
  {
    fprintf(stderr, "%s:\n  Link ID %d is disconnected\n",
            __PRETTY_FUNCTION__, _id);
    rc = -FI_ENOTCONN;
  }
  else                                 // Notify of unexpected error
  {
    if (rc == -FI_EAVAIL)              // Return actual error instead of EAVAIL
      rc = dumpCqError(_id, cq, __PRETTY_FUNCTION__);
    else
      fprintf(stderr, "%s:\n  No CQ entries for ID %d: rc %d: %s\n",
              __PRETTY_FUNCTION__, _id, rc, cq->error());
  }
  return rc;
}

ssize_t Pds::Eb::EbLfLink::postCompRecv(unsigned count)
{
  ssize_t rc = 0;

  // The number of consumable credits (count) must be
  // less than or equal to the number of credits available
  if (count > _credits)
  {
    fprintf(stderr, "%s:\n  Error: Link ID %d credits consummed (%u) > available (%u)\n",
            __PRETTY_FUNCTION__, _id, count, _credits);
    count = _credits;
  }

  _credits -= count;                    // Consume credits

  // Replenish credits
  unsigned retries = 0;
  while (_credits < _depth)
  {
    if ((rc = _ep->recv_comp_data(this)) < 0)
    {
      if ((rc != -FI_EAGAIN) || (++retries == 100000))
        break;
      else
        usleep(10);
    }
    ++_credits;
    retries = 0;
  }

  if ((rc < 0) && (rc != -FI_EAGAIN))
    fprintf(stderr, "%s:\n  Link ID %d failed to post a CQ buffer: %s\n",
            __PRETTY_FUNCTION__, _id, _ep->error());

  if (_credits != _depth)
    fprintf(stderr, "%s:\n  Error: Link ID %d credits: %u, max: %u, count: %u\n",
            __PRETTY_FUNCTION__, _id, _credits, _depth, count);

  return rc != -FI_EAGAIN ? rc : 0;
}
