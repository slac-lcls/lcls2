#ifndef Pds_Eb_EbLfLink_hh
#define Pds_Eb_EbLfLink_hh

#include "Endpoint.hh"

#include <stdint.h>
#include <cstddef>
#include <vector>


namespace Pds {
  namespace Eb {

    class EbLfLink
    {
    public:
      EbLfLink(Fabrics::Endpoint*, int depth, const unsigned& verbose);
    public:
      int recvU32(uint32_t* u32, const char* name);
      int sendU32(uint32_t  u32, const char* name);
      int sendMr(Fabrics::MemoryRegion*);
      int recvMr(Fabrics::RemoteAddress&);
      int setupMr(void* region, size_t size, Fabrics::MemoryRegion**);
    public:
      void*     lclAdx(size_t offset) const;
      uintptr_t rmtAdx(size_t offset) const;
    public:
      Fabrics::Endpoint* endpoint() const { return _ep;  }
      unsigned           id()       const { return _id;  }
    public:
      int post(const void* buf,
               size_t      len,
               uint64_t    immData);
      int poll(uint64_t* data);
      int poll(uint64_t* data, int msTmo);
    public:
      ssize_t postCompRecv();
      ssize_t postCompRecv(unsigned count);
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
    };

    class EbLfSvrLink : public EbLfLink
    {
    public:
      EbLfSvrLink(Fabrics::Endpoint*, int rxDepth, const unsigned& verbose);
    public:
      int prepare(unsigned id);
      int prepare(unsigned id,
                  size_t*  size);
      int setupMr(void* region, size_t size);
    private:
      int _synchronizeBegin();
      int _synchronizeEnd();
    };

    class EbLfCltLink : public EbLfLink
    {
    public:
      EbLfCltLink(Fabrics::Endpoint*, int rxDepth, const unsigned& verbose, uint64_t& pending);
    public:
      int prepare(unsigned id);
      int prepare(unsigned id,
                  void*    region,
                  size_t   lclSize,
                  size_t   rmtSize);
      int prepare(unsigned id,
                  void*    region,
                  size_t   size);
    public:
      int post(const void* buf,
               size_t      len,
               uint64_t    offset,
               uint64_t    immData,
               void*       ctx = nullptr);
    private:
      int _synchronizeBegin();
      int _synchronizeEnd();
    private:                           // Arranged in order of access frequency
      uint64_t& _pending;              // Bit list of IDs currently posting
    };
  };
};

inline
void* Pds::Eb::EbLfLink::lclAdx(size_t offset) const
{
  return static_cast<char*>(_mr->start()) + offset;
}

inline
uintptr_t Pds::Eb::EbLfLink::rmtAdx(size_t offset) const
{
  return _ra.addr + offset;
}

inline
ssize_t Pds::Eb::EbLfLink::postCompRecv(unsigned count)
{
  ssize_t rc = 0;

  for (unsigned i = 0; i < count; ++i)
  {
    rc = postCompRecv();
    if (rc)  break;
  }

  return rc;
}

#endif
