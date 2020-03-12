#ifndef HSD_Fmc134Ctrl_hh
#define HSD_Fmc134Ctrl_hh

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
      vuint32_t info;
      vuint32_t xcvr;
      vuint32_t status;
      vuint32_t adc_val;
      vuint32_t scramble;
      vuint32_t sw_trigger;
      vuint32_t lmfc_cnt;
      vuint32_t align_char;
      vuint32_t adc_pins;
      vuint32_t adc_pins_r;
      vuint32_t test_clksel;
      vuint32_t test_clkfrq;
    };
  };
};

#endif
