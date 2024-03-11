#ifndef HSD_PhaseMsmt_hh
#define HSD_PhaseMsmt_hh

#include "psdaq/mmhw/Reg.hh"
#include "Globals.hh"

namespace Pds {
  namespace HSD {
    class PhaseMsmt {
    public:
      Mmhw::Reg phaseA_even;
      Mmhw::Reg phaseA_odd;
      Mmhw::Reg phaseB_even;
      Mmhw::Reg phaseB_odd;
      Mmhw::Reg countA_even;
      Mmhw::Reg countA_odd;
      Mmhw::Reg countB_even;
      Mmhw::Reg countB_odd;
    };
  };
};

#endif
