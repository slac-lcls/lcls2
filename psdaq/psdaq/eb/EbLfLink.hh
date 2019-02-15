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
      EbLfLink(Fabrics::Endpoint*, unsigned verbose, uint64_t& pending);
      EbLfLink(Fabrics::Endpoint*, int rxDepth, unsigned verbose, uint64_t& pending);
      ~EbLfLink();
    public:
      int       preparePender(unsigned id);
      int       preparePender(unsigned id,
                              size_t*  size);
      int       preparePoster(unsigned id);
      int       preparePoster(unsigned id,
                              size_t   size);
      int       preparePoster(unsigned id,
                              void*    region,
                              size_t   lclSize,
                              size_t   rmtSize);
      int       preparePoster(unsigned id,
                              void*    region,
                              size_t   size);
    public:
      int       setupMr(void* region, size_t size);
      int       setupMr(void* region, size_t size, Fabrics::MemoryRegion**);
      int       recvU32(Fabrics::MemoryRegion*, uint32_t* u32, const char* name);
      int       sendU32(Fabrics::MemoryRegion*, uint32_t  u32, const char* name);
      int       sendMr(Fabrics::MemoryRegion*);
      int       recvMr(Fabrics::MemoryRegion*);
    public:
      void*     lclAdx(size_t offset) const;
      uintptr_t rmtAdx(size_t offset) const;
      int       postCompRecv(void* ctx = NULL);
      int       post(const void* buf,
                     size_t      len,
                     uint64_t    offset,
                     uint64_t    immData,
                     void*       ctx = nullptr);
      int       post(const void* buf,
                     size_t      len,
                     uint64_t    immData);
    public:
      Fabrics::Endpoint* endpoint() const { return _ep;  }
      unsigned           id()       const { return _id;  }
    private:
      int      _postCompRecv(unsigned count, void* ctx = NULL);
      int      _tryCq(fi_cq_data_entry*);
    private:                            // Arranged in order of access frequency
      Fabrics::Endpoint*      _ep;      // Endpoint
      Fabrics::MemoryRegion*  _mr;      // Memory Region
      Fabrics::RemoteAddress  _ra;      // Remote address descriptor
      int                     _rxDepth; // Depth  of the Rx Completion Queue
      int                     _rOuts;   // Number of completion buffers remaining
      uint64_t&               _pending; // Bit list of IDs currently posting
      unsigned                _id;      // ID     of peer on the remote side
      unsigned                _verbose; // Print some stuff if set
      char*                   _region;  // Used when App doesn't provide an MR
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

#endif
