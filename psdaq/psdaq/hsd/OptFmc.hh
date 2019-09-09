#ifndef OptFmc_hh
#define OptFmc_hh

#include "Globals.hh"

namespace Pds {
  namespace HSD {
    class OptFmc {
    public:
      vuint32_t fmc;
      vuint32_t qsfp;
      vuint32_t clks[7];
      vuint32_t adcOutOfRange[10];
      vuint32_t phaseCount_0;
      vuint32_t phaseValue_0;
      vuint32_t phaseCount_1;
      vuint32_t phaseValue_1;
    public:
      void resetPgp();
    };
  };
};

#endif
