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
      EbLfLink(Fabrics::Endpoint*, unsigned verbose);
    public:
      int setupMr(void* region, size_t size);
      int setupMr(void* region, size_t size, Fabrics::MemoryRegion**);
      int recvU32(uint32_t* u32, const char* name);
      int sendU32(uint32_t  u32, const char* name);
      int sendMr(Fabrics::MemoryRegion*);
      int recvMr(Fabrics::RemoteAddress&);
    public:
      void*     lclAdx(size_t offset) const;
      uintptr_t rmtAdx(size_t offset) const;
    public:
      Fabrics::Endpoint* endpoint() const { return _ep;  }
      unsigned           id()       const { return _id;  }
    protected:                         // Arranged in order of access frequency
      Fabrics::Endpoint*     _ep;      // Endpoint
      Fabrics::MemoryRegion* _mr;      // Memory Region
      Fabrics::RemoteAddress _ra;      // Remote address descriptor
      unsigned               _id;      // ID of peer
      const unsigned         _verbose; // Print some stuff if set
      char                   _buffer[sizeof(Fabrics::RemoteAddress)]; // Used when App doesn't provide an MR
      Fabrics::MemoryRegion* _bufMr;   // _buffer's
    };

    class EbLfSvrLink : public EbLfLink
    {
    public:
      EbLfSvrLink(Fabrics::Endpoint*, int rxDepth, unsigned verbose);
    public:
      int exchangeIds(unsigned id);
      int prepare();
      int prepare(size_t* size);
    public:
      int postCompRecv();
    private:
      int _postCompRecv(int count);
      int _tryCq(fi_cq_data_entry*);
    private:                          // Arranged in order of access frequency
      const int _depth;               // Depth  of the Completion Queue
      int       _count;               // Number of completion buffers remaining
    };

    class EbLfCltLink : public EbLfLink
    {
    public:
      EbLfCltLink(Fabrics::Endpoint*, unsigned verbose, uint64_t& pending);
    public:
      int exchangeIds(unsigned id);
      int prepare();
      int prepare(void*  region,
                  size_t lclSize,
                  size_t rmtSize);
      int prepare(void*  region,
                  size_t size);
    public:
      int post(const void* buf,
               size_t      len,
               uint64_t    offset,
               uint64_t    immData,
               void*       ctx = nullptr);
      int post(const void* buf,
               size_t      len,
               uint64_t    immData);
    private:                          // Arranged in order of access frequency
      uint64_t&    _pending;          // Bit list of IDs currently posting
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
int Pds::Eb::EbLfSvrLink::postCompRecv()
{
  if (--_count < 1)
  {
    _count += _postCompRecv(_depth - _count);
    if (_count < _depth)  return _depth - _count;
  }

  return 0;
}

#endif
