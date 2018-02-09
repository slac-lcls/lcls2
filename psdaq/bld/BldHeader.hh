#ifndef BldHeader_hh
#define BldHeader_hh

#include <stdio.h>

namespace Bld {
  class BldHeader {
  public:
    enum { sizeofFirst = 20, sizeofNext = 4 };
    BldHeader(uint64_t pulseId, uint64_t timeStamp, unsigned src)
    {
      uint64_t* p = reinterpret_cast<uint64_t*>(this);
      p[0] = pulseId;
      p[1] = timeStamp;
      *reinterpret_cast<uint32_t*>(&p[2]) = src;
    }
    BldHeader(uint64_t pulseId, uint64_t timeStamp, const BldHeader& ref)
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
  private:
  };
};

#endif
