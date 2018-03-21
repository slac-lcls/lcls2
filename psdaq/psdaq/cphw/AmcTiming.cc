#include "psdaq/cphw/AmcTiming.hh"

#include <stdio.h>
#include <unistd.h>

using namespace Pds::Cphw;

void AmcTiming::setPolarity(bool inverted)
{
  unsigned v = CSR;
  if (!inverted)
    v &= ~(1<<2);
  else
    v |= (1<<2);                        // Inverted
  CSR = v;
}

void AmcTiming::setLCLS()
{
  unsigned v = CSR;
  v &= ~(1<<4);
  CSR = v;
}

void AmcTiming::setLCLSII()
{
  unsigned v = CSR;
  v |= (1<<4);
  CSR = v;
}

void AmcTiming::bbReset()
{
  unsigned v = CSR;
  v |= (1<<3);
  CSR = v;
  usleep(10);
  v &= ~(1<<3);
  CSR = v;
}

void AmcTiming::resetStats()
{
  unsigned v = CSR;
  v |= (1<<0);
  CSR = v;
  usleep(10);
  v &= ~(1<<0);
  CSR = v;
}

bool AmcTiming::linkUp() const {
  unsigned v = CSR;
  return v & (1<<1);
}

void AmcTiming::dumpStats() const
{
#define PR(r) printf("%10.10s: 0x%x\n",#r,unsigned(r))

  PR(SOFcounts);
  PR(EOFcounts);
  PR(Msgcounts);
  PR(CRCerrors);
  PR(RxRecClks);
  PR(RxRstDone);
  PR(RxDecErrs);
  PR(RxDspErrs);
  { unsigned v = CSR;
    printf("%10.10s: 0x%x", "CSR", v);
    printf(" %s", v&(1<<1) ? "LinkUp":"LinkDn");
    if (v&(1<<2)) printf(" RXPOL");
    if (v&(1<<3)) printf(" RXRST");
    printf(" %s", v&(1<<4) ? "LCLSII":"LCLS");
    if (v&(1<<6)) printf(" BBRST");
    if (v&(1<<7)) printf(" PLLRST");
    if (v&(1<<8)) printf(" VSNERR");
    printf("\n");
  }
  PR(MsgDelay);
  PR(TxRefClks);
  PR(BuffByCnts);
}

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

