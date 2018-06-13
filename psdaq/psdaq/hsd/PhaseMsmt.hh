#ifndef HSD_PhaseMsmt_hh
#define HSD_PhaseMsmt_hh

#include <stdint.h>

namespace Pds {
  namespace HSD {
    class PhaseMsmt {
    public:
      uint32_t _evenPulse;
      uint32_t _oddPulse;
    };
  };
};

#endif
