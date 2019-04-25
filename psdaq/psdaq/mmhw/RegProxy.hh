#ifndef Pds_Mmhw_RegProxy_hh
#define Pds_Mmhw_RegProxy_hh

#include <stdint.h>

namespace Pds {
  namespace Mmhw {
    class RegProxy {
    public:
      static void initialize(void* base,
                             void* csr);
    public:
      RegProxy& operator=(const unsigned);
      operator unsigned() const;
    private:
      uint32_t _reserved;
    };
  };
};

#endif
