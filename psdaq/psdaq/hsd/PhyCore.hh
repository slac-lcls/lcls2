#ifndef HSD_PhyCore_hh
#define HSD_PhyCore_hh

#include "psdaq/mmhw/Reg.hh"
#include "Globals.hh"

namespace Pds {
  namespace HSD {

    class PhyCore {
    public:
      void dump() const;
    public:
      Mmhw::Reg rsvd_0[0x130/4];
      Mmhw::Reg bridgeInfo;
      Mmhw::Reg bridgeCSR;
      Mmhw::Reg irqDecode;
      Mmhw::Reg irqMask;
      Mmhw::Reg busLocation;
      Mmhw::Reg phyCSR;
      Mmhw::Reg rootCSR;
      Mmhw::Reg rootMSI1;
      Mmhw::Reg rootMSI2;
      Mmhw::Reg rootErrorFifo;
      Mmhw::Reg rootIrqFifo1;
      Mmhw::Reg rootIrqFifo2;
      Mmhw::Reg rsvd_160[2];
      Mmhw::Reg cfgControl;
      Mmhw::Reg rsvd_16c[(0x208-0x16c)/4];
      Mmhw::Reg barCfg[0x30/4];
      Mmhw::Reg rsvd_238[(0x1000-0x238)/4];
    };
  };
};

#endif
