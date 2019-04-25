#ifndef HSD_ClkSynth_hh
#define HSD_ClkSynth_hh

#include "psdaq/hsd/Globals.hh"
#include "psdaq/mmhw/RegProxy.hh"
#include <stdint.h>
#include <stdio.h>

namespace Pds {
  namespace HSD {
    class ClkSynth {
    public:
      void dump () const;
      void setup(TimingType);
    public:
      //      volatile uint32_t _reg[256];
      Pds::Mmhw::RegProxy _reg[256];
    };
  };
};

#endif
