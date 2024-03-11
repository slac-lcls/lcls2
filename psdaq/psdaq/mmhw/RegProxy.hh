#ifndef Pds_Mmhw_RegProxy_hh
#define Pds_Mmhw_RegProxy_hh

#include "psdaq/mmhw/Reg.hh"
#include <stdint.h>

namespace Pds {
  namespace Mmhw {
    class RegProxy {
    public:
      static void initialize(void* base,
                             void* csr,
                             bool  verbose=false);
    public:
      RegProxy& operator=(const unsigned);
      operator unsigned() const;
    private:
      Reg _reserved;
    };
  };
};

#endif
