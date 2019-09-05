#ifndef HSD_PhaseMsmt_hh
#define HSD_PhaseMsmt_hh

#include <stdint.h>

namespace Pds {
  namespace HSD {
    class PhaseMsmt {
    public:
      uint32_t phaseA_even;
      uint32_t phaseA_odd;
      uint32_t phaseB_even;
      uint32_t phaseB_odd;
      uint32_t countA_even;
      uint32_t countA_odd;
      uint32_t countB_even;
      uint32_t countB_odd;
    };
  };
};

#endif
