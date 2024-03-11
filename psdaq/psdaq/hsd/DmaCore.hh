#ifndef HSD_DmaCore_hh
#define HSD_DmaCore_hh

#include "psdaq/mmhw/Reg.hh"
#include "Globals.hh"

namespace Pds {
  namespace HSD {

    class DmaCore {
    public:
      void init(unsigned maxDmaSize=0);
      void dump() const;
      Mmhw::Reg rxEnable;
      Mmhw::Reg txEnable;
      Mmhw::Reg fifoClear;
      Mmhw::Reg irqEnable;
      Mmhw::Reg fifoValid; // W fifoThres, R b0 = inbound, b1=outbound
      Mmhw::Reg maxRxSize; // inbound
      Mmhw::Reg mode;      // b0 = online, b1=acknowledge, b2=ibrewritehdr
      Mmhw::Reg irqStatus; // W b0=ack, R b0=ibPend, R b1=obPend
      Mmhw::Reg irqRequests;
      Mmhw::Reg irqAcks;
      Mmhw::Reg irqHoldoff;
      Mmhw::Reg dmaCount;

      Mmhw::Reg reserved[244];

      Mmhw::Reg ibFifoPop;
      Mmhw::Reg obFifoPop;
      Mmhw::Reg reserved_pop[62];

      Mmhw::Reg loopFifoData; // RO
      Mmhw::Reg reserved_loop[63];

      Mmhw::Reg ibFifoPush[16];  // W data, R[0] status
      Mmhw::Reg obFifoPush[16];  // R b0=full, R b1=almost full, R b2=prog full
      Mmhw::Reg reserved_push[32];
    };
  };
};

#endif
