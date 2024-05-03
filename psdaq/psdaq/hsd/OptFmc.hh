#ifndef OptFmc_hh
#define OptFmc_hh

#include "psdaq/mmhw/Reg.hh"
#include "Globals.hh"

namespace Pds {
  namespace HSD {
    class OptFmc {
    public:
      Mmhw::Reg fmc;
      Mmhw::Reg qsfp;
      Mmhw::Reg clks[7];
      Mmhw::Reg adcOutOfRange[10];
      Mmhw::Reg phaseCount_0;
      Mmhw::Reg phaseValue_0;
      Mmhw::Reg phaseCount_1;
      Mmhw::Reg phaseValue_1;
      Mmhw::Reg txId;
    public:
      void resetPgp();
    };
  };
};

#endif
