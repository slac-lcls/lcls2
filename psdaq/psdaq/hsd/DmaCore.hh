#ifndef HSD_DmaCore_hh
#define HSD_DmaCore_hh

#include "Globals.hh"

namespace Pds {
  namespace HSD {

    class DmaCore {
    public:
      void init(unsigned maxDmaSize=0);
      void dump() const;
      vuint32_t rxEnable;
      vuint32_t txEnable;
      vuint32_t fifoClear;
      vuint32_t irqEnable;
      vuint32_t fifoValid; // W fifoThres, R b0 = inbound, b1=outbound
      vuint32_t maxRxSize; // inbound
      vuint32_t mode;      // b0 = online, b1=acknowledge, b2=ibrewritehdr
      vuint32_t irqStatus; // W b0=ack, R b0=ibPend, R b1=obPend
      vuint32_t irqRequests;
      vuint32_t irqAcks;
      vuint32_t irqHoldoff;
      vuint32_t dmaCount;

      vuint32_t reserved[244];

      vuint32_t ibFifoPop;
      vuint32_t obFifoPop;
      vuint32_t reserved_pop[62];

      vuint32_t loopFifoData; // RO
      vuint32_t reserved_loop[63];

      vuint32_t ibFifoPush[16];  // W data, R[0] status
      vuint32_t obFifoPush[16];  // R b0=full, R b1=almost full, R b2=prog full
      vuint32_t reserved_push[32];
    };
  };
};

#endif
