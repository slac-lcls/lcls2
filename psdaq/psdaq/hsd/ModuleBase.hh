#ifndef HSD_ModuleBase_hh
#define HSD_ModuleBase_hh

#include "TprCore.hh"
#include "DmaCore.hh"
#include "PhyCore.hh"
#include "FlashController.hh"

#include "psdaq/mmhw/AxiVersion.hh"
#include "psdaq/mmhw/RegProxy.hh"
#include "psdaq/mmhw/RingBuffer.hh"
#include "psdaq/mmhw/Xvc.hh"

namespace Pds {
  namespace HSD {
    class ModuleBase {
    public:
      void setRxAlignTarget(unsigned);
      void setRxResetLength(unsigned);
      void dumpRxAlign     () const;
    public:
      static unsigned local_id(unsigned bus); 
    public:
      Mmhw::AxiVersion version;
      uint32_t rsvd_to_0x08000[(0x8000-sizeof(version))/4];

      FlashController      flash;
      uint32_t rsvd_to_0x10000[(0x8000-sizeof(flash))/4];

      uint32_t i2c_regs[0x8000/4];  // FMC-dependent
      uint32_t regProxy[(0x08000)/4]; // 0x18000

      // DMA
      DmaCore           dma_core; // 0x20000
      uint32_t rsvd_to_0x30000[(0x10000-sizeof(dma_core))/4];

      // PHY
      PhyCore           phy_core; // 0x30000
      uint32_t rsvd_to_0x31000[(0x1000-sizeof(phy_core))/4];

      // GTH
      uint32_t gthAlign[10];     // 0x31000
      uint32_t rsvd_to_0x31100  [54];
      uint32_t gthAlignTarget;
      uint32_t gthAlignLast;
      uint32_t rsvd_to_0x31800[(0x800-0x108)/4];
      uint32_t gthDrp[0x200];

      // XVC
      Mmhw::Jtag    xvc;
      uint32_t rsvd_to_0x40000[(0xE000-sizeof(xvc))/4];

      // Timing
      TprCore  tpr;     // 0x40000
      uint32_t rsvd_to_0x50000  [(0x10000-sizeof(tpr))/4];

      Mmhw::RingBuffer         ring0;   // 0x50000
      uint32_t rsvd_to_0x60000  [(0x10000-sizeof(ring0))/4];

      Mmhw::RingBuffer         ring1;   // 0x60000
      uint32_t rsvd_to_0x70000  [(0x10000-sizeof(ring1))/4];
      uint32_t rsvd_to_0x80000  [0x10000/4];
    };
  };
};

#endif
