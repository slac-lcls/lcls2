#ifndef Pds_Bld_Header_hh
#define Pds_Bld_Header_hh

#include <stdio.h>

namespace Pds {
  namespace Bld {
    class Header {
    public:
      enum { sizeofFirst = 20, sizeofNext = 4 };
      enum { MTU = 8192 };
      Header() {}
      Header(uint64_t pulseId, uint64_t timeStamp, unsigned src)
      {
        uint64_t* p = reinterpret_cast<uint64_t*>(this);
        p[0] = pulseId;
        p[1] = timeStamp;
        *reinterpret_cast<uint32_t*>(&p[2]) = src;
      }
      Header(uint64_t pulseId, uint64_t timeStamp, const Header& ref)
      {
        uint32_t* p = reinterpret_cast<uint32_t*>(this);
        const uint64_t* q = reinterpret_cast<const uint64_t*>(&ref);
        *p  = ((pulseId   - q[0])&0xfff)<<20;
        *p |= (timeStamp - q[1])&0xfffff;
      }
    public:
      bool done(uint64_t pulseId) const
      {
        return (pulseId - *reinterpret_cast<const uint64_t*>(this) > 1023);
      }
      uint64_t pulseId() const
      {
        return reinterpret_cast<const uint64_t*>(this)[0];
      }
      unsigned id() const
      {
        return reinterpret_cast<const unsigned*>(this)[4];
      }
    private:
    };
  };
};

#endif
