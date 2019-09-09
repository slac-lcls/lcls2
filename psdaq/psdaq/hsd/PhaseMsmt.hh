#ifndef HSD_PhaseMsmt_hh
#define HSD_PhaseMsmt_hh

#include "Globals.hh"

namespace Pds {
  namespace HSD {
    class PhaseMsmt {
    public:
      vuint32_t phaseA_even;
      vuint32_t phaseA_odd;
      vuint32_t phaseB_even;
      vuint32_t phaseB_odd;
      vuint32_t countA_even;
      vuint32_t countA_odd;
      vuint32_t countB_even;
      vuint32_t countB_odd;
    };
  };
};

#endif
