#include "psdaq/hsd/Adt7411.hh"
#include <stdio.h>
#include <unistd.h>
#include <string.h>

using namespace Pds::HSD;

#define REG(r) (_reg[r]&0xff)

unsigned Adt7411::deviceId       () const { return REG(0x4d); }
unsigned Adt7411::manufacturerId () const { return REG(0x4e); }
unsigned Adt7411::interruptStatus() const { return REG(0x00) & (REG(0x01)<<8); }
unsigned Adt7411::interruptMask  () const { return REG(0x1d) & (REG(0x1e)<<8); }
unsigned Adt7411::internalTemp   () const { return REG(0x03) & (REG(0x07)<<8); }
unsigned Adt7411::externalTemp   () const { return REG(0x04) & (REG(0x08)<<8); }

void     Adt7411::start()
{
  _reg[0x18] = 0x9;  // start conversions
  _reg[0x19] = 0x0;
}

Adt7411_Mon Adt7411::mon() 
{
  unsigned r[128];
  for(unsigned i=0; i<128; i++) {
    r[i]=REG(i);
    usleep(1000);
  }

  unsigned Vdd=0, Tint=0;
  unsigned ain[8];

  unsigned v = r[3];
  Tint |= (v>>0)&3;
  Vdd  |= (v>>2)&3;

  v = r[4];
  ain[0] = (v>>0)&3;
  ain[1] = (v>>2)&3;
  ain[2] = (v>>4)&3;
  ain[3] = (v>>6)&3;
  
  v = r[5];
  ain[4] = (v>>0)&3;
  ain[5] = (v>>2)&3;
  ain[6] = (v>>4)&3;
  ain[7] = (v>>6)&3;

  Vdd  |= (r[6]<<2)&0x3fc;
  Tint |= (r[7]<<2)&0x3fc;
  for(unsigned i=0; i<8; i++) {
    v = r[8+i];
    ain[i] |= (v<<2)&0x3fc;
  }

  Adt7411_Mon m;
  m.Tint = double(int(Tint<<22))*0.25/double(1<<22)+40;
  m.Vdd  = double(Vdd)*3.11*2.197e-3;
  for(unsigned i=0; i<8; i++)
    m.ain[i] = double(ain[i])*2.197e-3;
  return m;
}

void     Adt7411::dump           ()
{
  start();

  usleep(500000);

  Adt7411_Mon m = mon();
  printf("Tint %6.2fC  Vdd %6.2fV\n",
         m.Tint,
         m.Vdd );
  for(unsigned i=0; i<4; i++)
    printf("  Ain[%u] %6.2fV",i,m.ain[i]);
  printf("\n");
  for(unsigned i=4; i<8; i++)
    printf("  Ain[%u] %6.2fV",i,m.ain[i]);
  printf("\n");
}

void Adt7411::interruptMask(unsigned m)
{
  _reg[0x1d] = m&0xff;
  _reg[0x1e] = (m>>8)&0xff;
}
