#ifndef Pds_Eb_EbLfLink_hh
#define Pds_Eb_EbLfLink_hh

#include "Endpoint.hh"

#include <stdint.h>
#include <cstddef>
#include <vector>


namespace Pds {
  namespace Eb {

    int setupMr(Fabrics::Fabric*        fabric,
                void*                   region,
                size_t                  size,
                Fabrics::MemoryRegion** mr,
                const unsigned&         verbose);

    class EbLfLink
    {
    public:
      EbLfLink(Fabrics::Endpoint*,
               const unsigned     depth,
               const unsigned&    verbose,
               volatile uint64_t& pending,
               volatile uint64_t& posting);
      ~EbLfLink();
    public:
      int recvU32(uint32_t* u32, const char* peer, const char* name);
      int sendU32(uint32_t  u32, const char* peer, const char* name);
      int sendMr(Fabrics::MemoryRegion*,  const char* peer);
      int recvMr(Fabrics::RemoteAddress&, const char* peer);
    public:
      void*     lclAdx(size_t      offset) const;
      size_t    lclOfs(const void* buffer) const;
      uintptr_t rmtAdx(size_t      offset) const;
      size_t    rmtOfs(uintptr_t   buffer) const;
    public:
      Fabrics::Endpoint* endpoint() const { return _ep;  }
      unsigned           id()       const { return _id;  }
      const uint64_t&    tmoCnt()   const { return _timedOut; }
      const uint64_t&    postCnt()  const { return _posted; }
      const uint64_t&    compCnt()  const { return _completed; }
      const uint64_t&    errCnt()   const { return _txErrors; }
    public:
      int post(const void* buf,
               size_t      len,
               uint64_t    immData);
      int post(uint64_t    immData);
      int poll(uint64_t* data);
      int poll(uint64_t* data, int msTmo);
    public:
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
      const unsigned         _depth;
      unsigned               _credits;
    };

    class EbLfSvrLink : public EbLfLink
    {
    public:
      EbLfSvrLink(Fabrics::Endpoint*,
                  const unsigned     rxDepth,
                  const unsigned&    verbose,
                  volatile uint64_t& pending,
                  volatile uint64_t& posting);
    public:
      int exchangeId(unsigned    id,
                     const char* peer);
      int prepare(size_t*     size,
                  const char* peer);
      int setupMr(void* region, size_t size, const char* peer);
    private:
      int _synchronizeBegin();
      int _synchronizeEnd();
    };

    class EbLfCltLink : public EbLfLink
    {
    public:
      EbLfCltLink(Fabrics::Endpoint*,
                  const unsigned     rxDepth,
                  const unsigned&    verbose,
                  volatile uint64_t& pending,
                  volatile uint64_t& posting);
    public:
      int exchangeId(unsigned    id,
                     const char* peer);
      int prepare(void*       region,
                  size_t      lclSize,
                  size_t      rmtSize,
                  const char* peer);
      int prepare(void*       region,
                  size_t      size,
                  const char* peer);
      int setupMr(void* region, size_t size);
    public:
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
