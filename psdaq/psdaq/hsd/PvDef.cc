#include "psdaq/hsd/PvDef.hh"
#include "psdaq/hsd/ChipAdcReg.hh"
#include <unistd.h>
#include <stdio.h>

using namespace Pds::HSD;

MonBufDetail::MonBufDetail(ChipAdcReg& reg, unsigned stream)
{
  for(unsigned j=0; j<16; j++) {
    reg.cacheSel = (stream<<4) | (j&0xf);
    usleep(1);
    unsigned state = reg.cacheState;
    unsigned addr  = reg.cacheAddr;
    bufstate[j] = (state>>0)&0xf;
    trgstate[j] = (state>>4)&0xf;
    bufbeg  [j] = float((addr >> 4)&0xfff) + 0.1*float((addr >> 0)&0xf);
    bufend  [j] = float((addr >>20)&0xfff) + 0.1*float((addr >>16)&0xf);
    //    printf("[%x] [%x] [%x]\n", reg.cacheSel, state, addr);
  }
}
