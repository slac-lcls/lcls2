#include "psdaq/hsd/DmaCore.hh"

#include <stdio.h>

using namespace Pds::HSD;

void DmaCore::init(unsigned maxDmaSize)
{
  //  Need to disable rx to reset dma size
  rxEnable = 0;
  if (maxDmaSize)
    maxRxSize = maxDmaSize;
  else
    maxRxSize = (1<<31);
  rxEnable = 1;
  
  fifoValid  = 254;
  //  irqHoldoff = 12500;  // 10kHz (100us)
  irqHoldoff = 0;
}

void DmaCore::dump() const 
{
#define PR(r) printf("%9.9s: %08x\n",#r, r)

  printf("DmaCore @ %p\n",this);
  PR(rxEnable);
  PR(txEnable);
  PR(fifoClear);
  PR(irqEnable);
  PR(fifoValid);
  PR(maxRxSize);
  PR(mode);
  PR(irqStatus);
  PR(irqRequests);
  PR(irqAcks);
  PR(irqHoldoff);
  PR(dmaCount);

#undef PR
}
