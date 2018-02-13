#ifndef Tpr_RxDesc_hh
#define Tpr_RxDesc_hh

#include <stdint.h>

namespace Pds {
  namespace Tpr {
    class RxDesc {
    public:
      RxDesc(uint32_t* d, unsigned sz) : maxSize(sz), data(d) {}
    public:
      uint32_t  maxSize;
      uint32_t* data; 
    };
  };
};

#endif

