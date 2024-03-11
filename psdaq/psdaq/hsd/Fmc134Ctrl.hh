#ifndef HSD_Fmc134Ctrl_hh
#define HSD_Fmc134Ctrl_hh

#include "psdaq/mmhw/Reg.hh"
#include "Globals.hh"

namespace Pds {
  namespace HSD {
    class Fmc134Cpld;
    class Fmc134Ctrl {
    public:
      void    remote_sync ();
      int32_t default_init(Fmc134Cpld&, unsigned mode=0);
      int32_t reset       ();
      void dump();
    public:
      Mmhw::Reg info;
      Mmhw::Reg xcvr;
      Mmhw::Reg status;
      Mmhw::Reg adc_val;
      Mmhw::Reg scramble;
      Mmhw::Reg sw_trigger;
      Mmhw::Reg lmfc_cnt;
      Mmhw::Reg align_char;
      Mmhw::Reg adc_pins;
      Mmhw::Reg adc_pins_r;
      Mmhw::Reg test_clksel;
      Mmhw::Reg test_clkfrq;
    };
  };
};

#endif
