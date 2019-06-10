#include "psdaq/hsd/Jesd204b.hh"
#include <unistd.h>
#include <stdio.h>

using namespace Pds::HSD;

Jesd204bStatus Jesd204b::status(unsigned i) const
{
  uint64_t r0 = reg[0x60+i];
  r0 <<= 32;
  r0 |= reg[0x10+i];

  Jesd204bStatus r;
  r.gtResetDone   = (r0>> 0)&1;
  r.recvDataValid = (r0>> 1)&1;
  r.dataNAlign    = (r0>> 2)&1;
  r.syncDone      = (r0>> 3)&1;
  r.bufOF         = (r0>> 4)&1;
  r.bufUF         = (r0>> 5)&1;
  r.commaNAlign   = (r0>> 6)&1;
  r.rxModEnable   = (r0>> 7)&1;
  r.sysRefDet     = (r0>> 8)&1;
  r.commaDet      = (r0>> 9)&1;
  r.dspErr        = (r0>>10)&0xff;
  r.decErr        = (r0>>18)&0xff;
  r.buffLatency   = (r0>>26)&0xff;
  r.cdrStatus     = (r0>>34)&1;
  return r;
}

void Jesd204b::clearErrors()
{
  uint32_t v = reg[4];
  reg[4] = v | (1<<3);
  usleep(10);
  reg[4] = v;
}

void Jesd204b::dumpStatus(const Jesd204bStatus* rxStatus, int n)
{
#define PRINTR(field) {                         \
    printf("%15.15s",#field);                   \
    for(unsigned i=0; i<16; i++)                \
      printf("%4x",rxStatus[i].field);          \
    printf("\n");                               \
  }

  printf("%15.15s","Lane");
  for(unsigned i=0; i<16; i++)
    printf("%4x",i);
  printf("\n");

  PRINTR(gtResetDone);
  PRINTR(recvDataValid);
  PRINTR(dataNAlign);
  PRINTR(syncDone);
  PRINTR(bufOF);
  PRINTR(bufUF);
  PRINTR(commaNAlign);
  PRINTR(rxModEnable);
  PRINTR(sysRefDet);
  PRINTR(commaDet);
  PRINTR(dspErr);
  PRINTR(decErr);
  PRINTR(buffLatency);
  PRINTR(cdrStatus);
}
