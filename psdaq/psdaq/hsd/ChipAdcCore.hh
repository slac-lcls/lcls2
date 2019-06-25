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
      uint32_t     rsvd0[0x800-sizeof(reg)/4];
      FexCfg       fex;
      uint32_t     rsvd1[0x800-sizeof(fex)/4];
    };
  };
};

#endif
