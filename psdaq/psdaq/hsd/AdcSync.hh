#ifndef HSD_AdcSync_hh
#define HSD_AdcSync_hh

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
      vuint32_t _cmd;
      //  cmd
      //  b0 = calibrate
      //  b15:1  = calib_time
      //  b19:16 = delay load
      mutable vuint32_t _select;
      //  select
      //  b1:0 = channel for readout
      //  b7:4 = word for readout
      vuint32_t _match;
      vuint32_t _rsvd;
      vuint32_t _delay[8];
      vuint32_t _rsvd2[0x1f4];
    };
  };
};

#endif
