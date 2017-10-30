#ifndef HSD_AdcSync_hh
#define HSD_AdcSync_hh

#include <stdint.h>

namespace Pds {
  namespace HSD {
    class AdcSync {
    public:
      void set_delay     (const uint32_t*);
      void start_training();
      void stop_training ();
      void dump_status   () const;
    private:
      uint32_t _cmd;
      //  cmd
      //  b0 = calibrate
      //  b15:1  = calib_time
      //  b19:16 = delay load
      mutable uint32_t _select;
      //  select
      //  b1:0 = channel for readout
      //  b7:4 = word for readout
      uint32_t _match;
      uint32_t _rsvd;
      uint32_t _delay[8];
      uint32_t _rsvd2[0xf4];
    };
  };
};

#endif
