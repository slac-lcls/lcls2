#ifndef HSD_PhyCore_hh
#define HSD_PhyCore_hh

#include "Globals.hh"

namespace Pds {
  namespace HSD {

    class PhyCore {
    public:
      void dump() const;
    public:
      vuint32_t rsvd_0[0x130/4];
      vuint32_t bridgeInfo;
      vuint32_t bridgeCSR;
      vuint32_t irqDecode;
      vuint32_t irqMask;
      vuint32_t busLocation;
      vuint32_t phyCSR;
      vuint32_t rootCSR;
      vuint32_t rootMSI1;
      vuint32_t rootMSI2;
      vuint32_t rootErrorFifo;
      vuint32_t rootIrqFifo1;
      vuint32_t rootIrqFifo2;
      vuint32_t rsvd_160[2];
      vuint32_t cfgControl;
      vuint32_t rsvd_16c[(0x208-0x16c)/4];
      vuint32_t barCfg[0x30/4];
      vuint32_t rsvd_238[(0x1000-0x238)/4];
    };
  };
};

#endif
