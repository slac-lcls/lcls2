#ifndef HSD_ChipAdcCore_hh
#define HSD_ChipAdcCore_hh

#include "ChipAdcReg.hh"
#include "FexCfg.hh"

#include <stdint.h>

namespace Pds {
  namespace HSD {
    class ChipAdcCore {
    public:
      ChipAdcReg   reg;
      uint32_t     rsvd0[(0x1000-sizeof(reg))>>2];
      FexCfg       fex;
      uint32_t     rsvd1[(0x1000-sizeof(fex))>>2];
    };
  };
};

#endif
