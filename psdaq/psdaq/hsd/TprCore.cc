#include "psdaq/hsd/TprCore.hh"

#include <stdio.h>
#include <unistd.h>
#include <time.h>

using namespace Pds::HSD;

bool TprCore::rxPolarity() const {
  volatile uint32_t v = CSR;
  return v&(1<<2);
}

void TprCore::rxPolarity(bool p) {
  volatile uint32_t v = CSR;
  v = p ? (v|(1<<2)) : (v&~(1<<2));
  CSR = v;
  usleep(10);
  CSR = v|(1<<3);
  usleep(10);
  CSR = v&~(1<<3);
}

void TprCore::resetRx() {
  volatile uint32_t v = CSR;
  CSR = (v|(1<<3));
  usleep(10);
  CSR = (v&~(1<<3));
}

void TprCore::resetRxPll() {
  volatile uint32_t v = CSR;
  CSR = (v|(1<<7));
  usleep(10);
  CSR = (v&~(1<<7));
}

void TprCore::resetBB() {
  volatile uint32_t v = CSR;
  CSR = (v|(1<<6));
  usleep(10);
  CSR = (v&~(1<<6));
}

void TprCore::resetCounts() {
  volatile uint32_t v = CSR;
  CSR = (v|1);
  usleep(10);
  CSR = (v&~1);
}

void TprCore::setLCLS() {
  volatile uint32_t v = CSR;
  CSR = v & ~(1<<4);
}

void TprCore::setLCLSII() {
  volatile uint32_t v = CSR;
  CSR = v | (1<<4);
}

static double clockRate(volatile uint32_t& clockReg)
{
  timespec tvb;
  clock_gettime(CLOCK_REALTIME,&tvb);
  volatile unsigned vvb = clockReg;

  usleep(10000);

  timespec tve;
  clock_gettime(CLOCK_REALTIME,&tve);
  unsigned vve = clockReg;
    
  double dt = double(tve.tv_sec-tvb.tv_sec)+1.e-9*(double(tve.tv_nsec)-double(tvb.tv_nsec));
  return 16.e-6*double(vve-vvb)/dt;
}

double TprCore::txRefClockRate() const { return clockRate(const_cast<volatile uint32_t&>(TxRefClks)); }
double TprCore::rxRecClockRate() const { return clockRate(const_cast<volatile uint32_t&>(RxRecClks)); }

void TprCore::dump() const {
  printf("SOFcounts: %08x\n", SOFcounts);
  printf("EOFcounts: %08x\n", EOFcounts);
  printf("Msgcounts: %08x\n", Msgcounts);
  printf("CRCerrors: %08x\n", CRCerrors);
  printf("RxRecClks: %08x\n", RxRecClks);
  printf("RxRstDone: %08x\n", RxRstDone);
  printf("RxDecErrs: %08x\n", RxDecErrs);
  printf("RxDspErrs: %08x\n", RxDspErrs);
  { unsigned v = CSR;
    printf("CSR      : %08x", v); 
    printf(" %s", v&(1<<1) ? "LinkUp":"LinkDn");
    if (v&(1<<2)) printf(" RXPOL");
    printf(" %s", v&(1<<4) ? "LCLSII":"LCLS");
    if (v&(1<<5)) printf(" LinkDnL");
    printf("\n");
    //  Acknowledge linkDownL bit
    const_cast<TprCore*>(this)->CSR = v & ~(1<<5);
  }
  printf("RxDspErrs: %08x\n", RxDspErrs);
  printf("TxRefClks: %08x\n", TxRefClks);
  printf("BypDone  : %04x\n", (BypassCnts>> 0)&0xffff);
  printf("BypResets: %04x\n", (BypassCnts>>16)&0xffff);
  printf("Version  : %08x\n", Version);
}
