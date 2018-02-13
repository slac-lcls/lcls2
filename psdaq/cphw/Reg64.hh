#ifndef Cphw_Reg64_hh
#define Cphw_Reg64_hh

#include "psdaq/cphw/Reg.hh"

namespace Pds {
  namespace Cphw {
    class Reg64 {
    public:
      Reg64& operator=(const uint64_t v) { low=unsigned(v&0xffffffff); high=unsigned(v>>32); return *this; }
      operator uint64_t() const { uint64_t v=unsigned(high); v<<=32; v|=unsigned(low); return v; }
      operator uint32_t() const { return unsigned(low); }
    public:
      Reg low;
      Reg high;
    };
  };
};

#endif
