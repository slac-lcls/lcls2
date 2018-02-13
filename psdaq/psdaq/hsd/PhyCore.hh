#ifndef HSD_PhyCore_hh
#define HSD_PhyCore_hh

#include <stdint.h>

namespace Pds {
  namespace HSD {

    class PhyCore {
    public:
      void dump() const;
    public:
      uint32_t rsvd_0[0x130/4];
      uint32_t bridgeInfo;
      uint32_t bridgeCSR;
      uint32_t irqDecode;
      uint32_t irqMask;
      uint32_t busLocation;
      uint32_t phyCSR;
      uint32_t rootCSR;
      uint32_t rootMSI1;
      uint32_t rootMSI2;
      uint32_t rootErrorFifo;
      uint32_t rootIrqFifo1;
      uint32_t rootIrqFifo2;
      uint32_t rsvd_160[2];
      uint32_t cfgControl;
      uint32_t rsvd_16c[(0x208-0x16c)/4];
      uint32_t barCfg[0x30/4];
      uint32_t rsvd_238[(0x1000-0x238)/4];
    };
  };
};

#endif
