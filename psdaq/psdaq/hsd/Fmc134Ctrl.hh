#ifndef HSD_Fmc134Ctrl_hh
#define HSD_Fmc134Ctrl_hh

#include <stdint.h>

namespace Pds {
  namespace HSD {

    class Fmc134Ctrl {
    public:
      void dump();
    public:
      uint32_t info;
      uint32_t xcvr;
      uint32_t status;
      uint32_t adc_val;
      uint32_t scramble;
      uint32_t sw_trigger;
      uint32_t lmfc_cnt;
      uint32_t align_char;
      uint32_t adc_pins;
      uint32_t adc_pins_r;
      uint32_t test_clksel;
      uint32_t test_clkfrq;
    };
  };
};

#endif
