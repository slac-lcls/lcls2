#ifndef HSD_AdcCore_hh
#define HSD_AdcCore_hh

#include "psdaq/mmhw/Reg.hh"
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
      Mmhw::Reg _cmd;
      Mmhw::Reg _status;
      Mmhw::Reg _master_start;
      Mmhw::Reg _adrclk_delay_set_auto;
      Mmhw::Reg _channel_select;
      Mmhw::Reg _tap_match_lo;
      Mmhw::Reg _tap_match_hi;
      Mmhw::Reg _adc_req_tap ;
      uint32_t  _rsvd2[0xf8];
    };
  };
};

#endif
