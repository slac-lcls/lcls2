#ifndef HSD_FmcCore_hh
#define HSD_FmcCore_hh

#include "psdaq/mmhw/Reg.hh"
#include "Globals.hh"

namespace Pds {
  namespace HSD {
    class FmcCore {
    public:
      bool   present  () const;
      bool   powerGood() const;
      void   selectClock(unsigned);
      double clockRate() const;
      void   cal_enable ();
      void   cal_disable();
    private:
      Mmhw::Reg _irq;
      Mmhw::Reg _irq_en;
      Mmhw::Reg _rsvd[6];
      Mmhw::Reg _detect;
      Mmhw::Reg _cmd;
      Mmhw::Reg _ctrl;
      Mmhw::Reg _rsvd2[5];
      Mmhw::Reg _clock_select;
      Mmhw::Reg _clock_count;
      Mmhw::Reg _rsvd3[0xee];
    };
  };
};

#endif
