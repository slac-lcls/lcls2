#include "psdaq/hsd/Tps2481.hh"
#include <stdio.h>
#include <unistd.h>

using namespace Pds::HSD;

void Tps2481::dump()
{
#define print_r(r) { printf("%s %02x  ",#r,_##r); usleep(1000); }
#define print_s(r,s,u) {                                \
    unsigned v = _##r;                                  \
      printf("%s %02x [%4.2f%s ",#r,v,double(v)*s,u);   \
      usleep(1000); }

  print_r(cfg);  
  print_s(shtv,0.01,"mV");  
  print_s(busv,0.004,"V");  
  print_r(pwr);  
  print_r(cur);  
  print_r(cal);  
  printf("\n");
}
