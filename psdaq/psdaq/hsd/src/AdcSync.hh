#ifndef HSD_AdcSync_hh
#define HSD_AdcSync_hh

#include "psdaq/mmhw/Reg.hh"
#include "Globals.hh"

namespace Pds {
  namespace HSD {
    class AdcSync {
    public:
      void set_delay     (const unsigned*);
      void start_training();
      void stop_training ();
      void dump_status   () const;
    private:
      Mmhw::Reg _cmd;
      //  cmd
      //  b0 = calibrate
      //  b15:1  = calib_time
      //  b19:16 = delay load
      mutable Mmhw::Reg _select;
      //  select
      //  b1:0 = channel for readout
      //  b7:4 = word for readout
      Mmhw::Reg _match;
      uint32_t  _rsvd;
      Mmhw::Reg _delay[8];
      uint32_t  _rsvd2[0x1f4];
    };
  };
};

#endif
