#include "psdaq/cphw/TimingRx.hh"

#include <stdio.h>
#include <unistd.h>

using namespace Pds::Cphw;

void TimingRx::setPolarity(bool inverted)
{
  unsigned v = CSR;
  if (!inverted)
    v &= ~(1<<2);
  else
    v |= (1<<2);                        // Inverted
  CSR = v;
}

void TimingRx::setLCLS()
{
  unsigned v = CSR;
  v &= ~(1<<4);
  CSR = v;
}

void TimingRx::setLCLSII()
{
  unsigned v = CSR;
  v |= (1<<4);
  CSR = v;
}

void TimingRx::bbReset()
{
  unsigned v = CSR;
  v |= (1<<3);
  CSR = v;
  usleep(10);
  v &= ~(1<<3);
  CSR = v;
}

void TimingRx::resetStats()
{
  unsigned v = CSR;
  v |= (1<<0);
  CSR = v;
  usleep(10);
  v &= ~(1<<0);
  CSR = v;
}

bool TimingRx::linkUp() const {
  unsigned v = CSR;
  return v & (1<<1);
}

void TimingRx::dumpStats() const
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

