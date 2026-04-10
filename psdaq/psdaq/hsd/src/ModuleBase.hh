#ifndef HSD_ModuleBase_hh
#define HSD_ModuleBase_hh

//#include "DmaCore.hh"
#include "PhyCore.hh"
//#include "FlashController.hh"

#include "psdaq/mmhw/RegProxy.hh"
#include "psdaq/mmhw/RingBuffer.hh"
#include "psdaq/mmhw/TprCore.hh"
#include "psdaq/mmhw/Xvc.hh"

typedef volatile uint32_t vuint32_t;

namespace Pds {
  namespace HSD {
    class ModuleBase {
    public:
      void setRxAlignTarget(unsigned);
      void setRxResetLength(unsigned);
      void dumpRxAlign     () const;
    public:
      static unsigned local_id(unsigned bus); 
      static ModuleBase* create(int);
    public:
      uint32_t  rsvd_to_0x10_0000[0x100000/4];
        
      // I2C (0010_0000)
      uint32_t  i2c_regs[0x8000/4];  // FMC-dependent
      uint32_t  regProxy[0x8000/4];  // 0x108000

      // GTH (0011_0000)
      Mmhw::Reg gthAlign[10];
      uint32_t  rsvd_to_0x11_0100  [54];
      Mmhw::Reg gthAlignTarget;
      Mmhw::Reg gthAlignLast;
      uint32_t  rsvd_to_0x11_4000[(0x4000-0x108)/4];
      Mmhw::Reg gthDrp[0x200];
      uint32_t  rsvd_to_0x14_0000[(0x2C000-sizeof(gthDrp))/4];

      // Timing (0014_0000)
      Mmhw::TprCore            tpr;
      uint32_t rsvd_to_0x15_0000  [(0x10000-sizeof(tpr))/4];

      Mmhw::RingBuffer         ring0;   // 0x15_0000
      uint32_t rsvd_to_0x16_0000  [(0x10000-sizeof(ring0))/4];

      Mmhw::RingBuffer         ring1;   // 0x16_0000
      uint32_t  rsvd_to_0x17_0000  [(0x10000-sizeof(ring1))/4];
        
      Mmhw::Reg tprLoopback;
    };
  };
};

#endif
