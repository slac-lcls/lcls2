#ifndef HSD_DmaCore_hh
#define HSD_DmaCore_hh

#include <stdint.h>

namespace Pds {
  namespace HSD {

    class DmaCore {
    public:
      void init(unsigned maxDmaSize=0);
      void dump() const;
      uint32_t rxEnable;
      uint32_t txEnable;
      uint32_t fifoClear;
      uint32_t irqEnable;
      uint32_t fifoValid; // W fifoThres, R b0 = inbound, b1=outbound
      uint32_t maxRxSize; // inbound
      uint32_t mode;      // b0 = online, b1=acknowledge, b2=ibrewritehdr
      uint32_t irqStatus; // W b0=ack, R b0=ibPend, R b1=obPend
      uint32_t irqRequests;
      uint32_t irqAcks;
      uint32_t irqHoldoff;
      uint32_t dmaCount;

      uint32_t reserved[244];

      uint32_t ibFifoPop;
      uint32_t obFifoPop;
      uint32_t reserved_pop[62];

      uint32_t loopFifoData; // RO
      uint32_t reserved_loop[63];

      uint32_t ibFifoPush[16];  // W data, R[0] status
      uint32_t obFifoPush[16];  // R b0=full, R b1=almost full, R b2=prog full
      uint32_t reserved_push[32];
    };
  };
};

#endif
