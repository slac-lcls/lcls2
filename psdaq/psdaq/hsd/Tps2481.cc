#include "psdaq/hsd/Tps2481.hh"
#include <stdio.h>
#include <unistd.h>

using namespace Pds::HSD;

static const double currLSB=20e-6;
static const double powerLSB=400e-6;

//#define SWAP(v) v
#define SWAP(v) ((v&0x00ff)<<8)|((v&0xff00)>>8)

double Tps2481::current_A() const
{
  uint16_t v = _cur; usleep(1000);
  v = SWAP(v);
  return double(v)*currLSB;
}

double Tps2481::power_W() const
{
  uint16_t v = _pwr; usleep(1000);
  v = SWAP(v);
  return double(v)*powerLSB;
}

void Tps2481::start()
{
  _cal = SWAP(0x1000);
}

void Tps2481::dump()
{
  start();
  usleep(1000);

#define print_r(r) { uint16_t v = _##r; printf("%s %04x  ",#r,SWAP(v)); usleep(1000); }
#define print_s(r,s,u) {                                \
    unsigned v = _##r;                                  \
      v = SWAP(v);                                      \
      printf("%s %04x [%4.2f%s] ",#r,v,double(v)*s,u);   \
      usleep(1000); }

  print_r(cfg);  
  print_s(shtv,0.01,"mV");  
  print_s(busv,0.0005,"V");  
  print_s(pwr, powerLSB,"W");  
  print_s(cur, currLSB, "A");  
  print_r(cal);  
  printf("\n");
}
