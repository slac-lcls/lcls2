#ifndef Pds_Eb_EbLfLink_hh
#define Pds_Eb_EbLfLink_hh

#include "psdaq/eb/Endpoint.hh"

#include <stdint.h>
#include <cstddef>
#include <vector>


namespace Pds {
  namespace Eb {

    class EbLfLink
    {
    public:
      EbLfLink(Fabrics::Endpoint*);
      EbLfLink(Fabrics::Endpoint*, int rxDepth);
      ~EbLfLink();
    public:
      int       preparePender(unsigned idx,
                              unsigned id,
                              unsigned verbose,
                              void*    ctx = nullptr);
      int       preparePender(void*    region,
                              size_t   size,
                              unsigned idx,
                              unsigned id,
                              unsigned verbose,
                              void*    ctx = nullptr);
      int       preparePoster(unsigned idx,
                              unsigned id,
                              unsigned verbose);
      int       preparePoster(void*    region,
                              size_t   size,
                              unsigned idx,
                              unsigned id,
                              unsigned verbose);
    public:
      int       setupMr(void* region, size_t size);
      int       recvId();
      int       sendId(unsigned idx, unsigned id);
      int       syncLclMr();
      int       syncRmtMr(size_t size);
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
      unsigned           index()    const { return _idx; }
      unsigned           id()       const { return _id;  }
    private:
      int       _postCompRecv(unsigned count, void* ctx = NULL);
      int       _tryCq(fi_cq_data_entry*);
    private:
      Fabrics::Endpoint*      _ep;      // Endpoint
      Fabrics::MemoryRegion*  _mr;      // Memory Region
      Fabrics::RemoteAddress  _ra;      // Remote address descriptor
      int                     _rxDepth; // Depth  of the Rx Completion Queue
      int                     _rOuts;   // Number of completion buffers remaining
      unsigned                _idx;     // Index  of peer on the remote side
      unsigned                _id;      // ID     of peer on the remote side
      char*                   _region;  // Used when App doesn't provide an MR
      unsigned                _verbose; // Print some stuff if set
    };
  };
};

inline
void* Pds::Eb::EbLfLink::lclAdx(size_t offset) const
{
  return (char*)_mr->start() + offset;
}

inline
uintptr_t Pds::Eb::EbLfLink::rmtAdx(size_t offset) const
{
  return _ra.addr + offset;
}

#endif
