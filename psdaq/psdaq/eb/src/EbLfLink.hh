/**
 * @file
 * @brief EbLfLink and its server and client variants: one libfabric connection used by the event builder, with credit-managed receives, immediate-data exchanges and RDMA posting.
 */
#ifndef Pds_Eb_EbLfLink_hh
#define Pds_Eb_EbLfLink_hh

#include "Endpoint.hh"

#include <stdint.h>
#include <cstddef>
#include <vector>


namespace Pds {
  namespace Eb {

    /** Make *mr a memory region on fabric that covers size bytes at region. An existing *mr that starts at region and is at least size bytes long is reused; otherwise it is deregistered and a new region is registered and stored in *mr (even on failure). Returns 0 or the fabric error number. */
    int setupMr(Fabrics::Fabric*        fabric,
                void*                   region,
                size_t                  size,
                Fabrics::MemoryRegion** mr,
                const unsigned&         verbose);

    /** One libfabric connection between an event-builder client and server. Small values are exchanged as immediate data, a fixed number of zero-length receives (credits) is kept posted, and addresses are translated to and from offsets in the local and remote memory regions. */
    class EbLfLink
    {
    public:
      /** Wrap the endpoint with depth receive credits, zero the shared pending and posting words and post depth zero-length receives. The peer ID starts as (unsigned)-1. */
      EbLfLink(Fabrics::Endpoint*,
               const unsigned     depth,
               const unsigned&    verbose,
               volatile uint64_t& pending,
               volatile uint64_t& posting);
      /** Zero the pending and posting words and print a message if fewer than depth credits are posted. The endpoint is not deleted. */
      ~EbLfLink();
    public:
      /** Wait up to 7 s for one immediate data value and store its low 32 bits in u32. Returns 0, or the poll() error after printing a message naming name and peer. */
      int recvU32(uint32_t* u32, const char* peer, const char* name);
      /** Send u32 as immediate data with post(); returns 0, or the error after printing a message. */
      int sendU32(uint32_t  u32, const char* peer, const char* name);
      /** Send a RemoteAddress built from the memory region (start address, length, key) to the peer as a series of 32-bit immediate data values; returns 0 or a negative error. */
      int sendMr(Fabrics::MemoryRegion*,  const char* peer);
      /** Receive a RemoteAddress into ra as a series of 32-bit immediate data values (as sent by sendMr()), waiting up to 7 s for each; returns 0 or the poll() error. */
      int recvMr(Fabrics::RemoteAddress&, const char* peer);
    public:
      /** Return the local address that is offset bytes from the start of the local memory region. */
      void*     lclAdx(size_t      offset) const;
      /** Return the byte offset of buffer from the start of the local memory region. */
      size_t    lclOfs(const void* buffer) const;
      /** Return the remote address that is offset bytes from the start of the remote region. */
      uintptr_t rmtAdx(size_t      offset) const;
      /** Return the byte offset of the remote address buffer from the start of the remote region. */
      size_t    rmtOfs(uintptr_t   buffer) const;
    public:
      /** Return the libfabric endpoint. */
      Fabrics::Endpoint* endpoint() const { return _ep;  }
      /** Return the peer ID received in exchangeId(); (unsigned)-1 before that. */
      unsigned           id()       const { return _id;  }
      /** Return the number of timeouts counted by post() and poll(). */
      const uint64_t&    tmoCnt()   const { return _timedOut; }
      const uint64_t&    postCnt()  const { return _posted; }
      const uint64_t&    compCnt()  const { return _completed; }
      const uint64_t&    errCnt()   const { return _txErrors; }
    public:
      /** Send len bytes from buf with immediate data immData using Endpoint::injectdata(), retrying -FI_EAGAIN for up to 300 s. This peer's bit is set in the posting word while sending. Returns 0 or the last error (a timeout is counted). */
      int post(const void* buf,
               size_t      len,
               uint64_t    immData);
      /** Send only the immediate data immData; same as post(nullptr, 0, immData). */
      int post(uint64_t    immData);
      /** Check the receive completion queue once without waiting and re-post the consumed receive. On a completion, store its immediate data in data and return 0; otherwise return -FI_EAGAIN (counted as a timeout), -FI_ENOTCONN if the peer has shut down, or another negative error. */
      int poll(uint64_t* data);
      /** Like poll(uint64_t*), but waits up to msTmo milliseconds for a completion. The shared pending word is incremented while waiting. */
      int poll(uint64_t* data, int msTmo);
    public:
      /** Consume count receive credits, then re-post zero-length receives until depth credits are posted again (on -FI_EAGAIN, retry up to 100000 times with 10 us sleeps). Returns 0 or a negative error other than -FI_EAGAIN. */
      ssize_t postCompRecv(const unsigned count);
    protected:
      // Drain the Tx CQ without blocking, so an RMA write that fails after
      // being queued is seen.  fi_writedata returning FI_SUCCESS means only
      // that the write was queued; the delivery status arrives here.  Also
      // required under FI_PROGRESS_MANUAL, where reading a CQ *is* what
      // advances the provider.
      //
      // Called by post(), which is the only place that knows a write has been
      // queued, so no caller has to remember to do it.  A no-op on a link with
      // no Tx CQ, which is every server-side link.
      ssize_t _reapTx();
    protected:
      enum { _BegSync = 0x11111111,
             _EndSync = 0x22222222,
             _SvrSync = 0x33333333,
             _CltSync = 0x44444444 };
    protected:                         // Arranged in order of access frequency
      unsigned               _id;      // ID of peer
      Fabrics::Endpoint*     _ep;      // Endpoint
      Fabrics::MemoryRegion* _mr;      // Memory Region
      Fabrics::RemoteAddress _ra;      // Remote address descriptor
      const unsigned&        _verbose; // Print some stuff if set
      uint64_t               _timedOut;
      volatile uint64_t&     _pending; // Flag set when currently pending
      volatile uint64_t&     _posting; // Bit list of IDs currently posting
      uint64_t               _posted;    // RMA writes queued
      uint64_t               _completed; // Their Tx completions reaped
      uint64_t               _txErrors;  // Of which reported an error
    public:
      const unsigned         _depth;  ///< Number of receive buffers (credits) to keep posted.
      unsigned               _credits;  ///< Number of receive buffers currently posted.
    };

    /** Server side of an EbLfLink; synchronizes with the client before and after each exchange. */
    class EbLfSvrLink : public EbLfLink
    {
    public:
      /** Construct the base EbLfLink with rxDepth receive credits. */
      EbLfSvrLink(Fabrics::Endpoint*,
                  const unsigned     rxDepth,
                  const unsigned&    verbose,
                  volatile uint64_t& pending,
                  volatile uint64_t& posting);
    public:
      /** Synchronize with the client, receive its ID (stored as id()), send id, then synchronize again. Returns 0 or the first error. */
      int exchangeId(unsigned    id,
                     const char* peer);
      /** Synchronize with the client and receive the size of the memory region it asks for, storing it in size if size is non-null. setupMr() completes this exchange. */
      int prepare(size_t*     size,
                  const char* peer);
      /** Register (or reuse) a memory region for size bytes at region, send its description to the client with sendMr() and finish the synchronization started by prepare(). Returns 0 or the first error. */
      int setupMr(void* region, size_t size, const char* peer);
    private:
      int _synchronizeBegin();
      int _synchronizeEnd();
    };

    /** Client side of an EbLfLink; writes data into the server's registered memory with RDMA. */
    class EbLfCltLink : public EbLfLink
    {
    public:
      /** Construct the base EbLfLink with rxDepth receive credits. */
      EbLfCltLink(Fabrics::Endpoint*,
                  const unsigned     rxDepth,
                  const unsigned&    verbose,
                  volatile uint64_t& pending,
                  volatile uint64_t& posting);
    public:
      /** Synchronize with the server, send id, receive the server's ID (stored as id()), then synchronize again. Returns 0 or the first error. */
      int exchangeId(unsigned    id,
                     const char* peer);
      /** Synchronize with the server. If region is non-null, send rmtSize, register (or reuse) a local memory region of lclSize bytes at region and receive the server's region description. Then synchronize again. Returns 0 or the first error. */
      int prepare(void*       region,
                  size_t      lclSize,
                  size_t      rmtSize,
                  const char* peer);
      /** Call prepare(region, size, size, peer). */
      int prepare(void*       region,
                  size_t      size,
                  const char* peer);
      /** Register (or reuse) a local memory region for size bytes at region with Pds::Eb::setupMr(); returns -1 if there is no endpoint. */
      int setupMr(void* region, size_t size);
    public:
      /** RDMA-write len bytes from buf to the server's region at offset, with immediate data immData (Endpoint::writedata()), retrying -FI_EAGAIN for up to 300 s. This peer's bit is set in the posting word while sending. Returns 0 or the last error. */
      int post(const void* buf,
               size_t      len,
               size_t      offset,
               uint64_t    immData,
               void*       ctx = nullptr);
    private:
      int _synchronizeBegin();
      int _synchronizeEnd();
    };
  };
};

inline
void* Pds::Eb::EbLfLink::lclAdx(size_t offset) const
{
  return static_cast<char*>(_mr->start()) + offset;
}

inline
size_t Pds::Eb::EbLfLink::lclOfs(const void* buffer) const
{
  return static_cast<const char*>(buffer) -
         static_cast<const char*>(_mr->start());
}

inline
uintptr_t Pds::Eb::EbLfLink::rmtAdx(size_t offset) const
{
  return _ra.addr + offset;
}

inline
size_t Pds::Eb::EbLfLink::rmtOfs(uintptr_t buffer) const
{
  return buffer - _ra.addr;
}

inline
int Pds::Eb::EbLfLink::post(uint64_t immData)
{
  return post(nullptr, 0, immData);
}

#endif
