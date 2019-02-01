#include "psdaq/cphw/AmcTiming.hh"

#include <stdio.h>
#include <unistd.h>

using namespace Pds::Cphw;

void AmcTiming::setRxAlignTarget(unsigned t)
{
  unsigned v = gthAlignTarget;
  v &= ~0x3f;
  v |= (t&0x3f);
  gthAlignTarget = v;
}

void AmcTiming::setRxResetLength(unsigned len)
{
  unsigned v = gthAlignTarget;
  v &= ~0xf0000;
  v |= (len&0xf)<<16;
  gthAlignTarget = v;
}

void AmcTiming::dumpRxAlign     () const
{
  printf("\nTarget: 0x%x\tMask: 0x%x\tRstLen: %u\tLast: 0x%x\n",
         gthAlignTarget&0x7f,
         (gthAlignTarget>>8)&0x7f,
         (gthAlignTarget>>16)&0xf,
         gthAlignLast&0x7f);
  for(unsigned i=0; i<40; i++) {
    printf(" %04x",(gthAlign[i/2] >> (16*(i&1)))&0xffff);
    if ((i%10)==9) printf("\n");
  }
  printf("\n");
}

