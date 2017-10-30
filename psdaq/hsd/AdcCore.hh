#ifndef HSD_AdcCore_hh
#define HSD_AdcCore_hh

#include <stdint.h>

namespace Pds {
  namespace HSD {
    class AdcCore {
    public:
      void init_training (unsigned);
      void start_training();
      void dump_training ();
      void loop_checking ();
      void capture_idelay();
      void pulse_sync    ();
      void set_ref_delay (unsigned);
      void dump_status   () const;
    private:
      uint32_t _cmd;
      uint32_t _status;
      uint32_t _master_start;
      uint32_t _adrclk_delay_set_auto;
      uint32_t _channel_select;
      uint32_t _tap_match_lo;
      uint32_t _tap_match_hi;
      uint32_t _adc_req_tap ;
      uint32_t _rsvd2[0xf8];
    };
  };
};

#endif
