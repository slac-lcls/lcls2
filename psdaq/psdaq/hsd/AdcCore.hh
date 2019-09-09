#ifndef HSD_AdcCore_hh
#define HSD_AdcCore_hh

#include "Globals.hh"

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
      vuint32_t _cmd;
      vuint32_t _status;
      vuint32_t _master_start;
      vuint32_t _adrclk_delay_set_auto;
      vuint32_t _channel_select;
      vuint32_t _tap_match_lo;
      vuint32_t _tap_match_hi;
      vuint32_t _adc_req_tap ;
      vuint32_t _rsvd2[0xf8];
    };
  };
};

#endif
