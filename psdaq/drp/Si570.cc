#include "Si570.hh"
#include "DataDriver.h"

#include "psalg/utils/SysLog.hh"
using logging = psalg::SysLog;

using namespace Drp;

Si570::Si570() {}
Si570::~Si570() {}

void Si570::reset()
{
  unsigned v = _reg[135];
  v |= 1;
  _reg[135] = 1;
  do { 
    usleep(100); 
    v = _reg[135];
  } while (v&1);
}

double Si570::read()
{
  //  Read factory calibration for 156.25 MHz
  static const unsigned hsd_divn[] = {4,5,6,7,0,9,0,11};
  unsigned v = _reg[7];
  logging::info("si570[7] = 0x%x\n", v);
  unsigned hs_div = hsd_divn[(v>>5)&7];
  unsigned n1 = (v&0x1f)<<2;
  v = _reg[8];
  logging::info("si570[8] = 0x%x\n", v);
  n1 |= (v>>6)&3;
  uint64_t rfreq = v&0x3f;
  for(unsigned i=9; i<13; i++) {
    v = _reg[i];
    logging::info("si570[%d] = 0x%x\n", i, v);
    rfreq <<= 8;
    rfreq |= (v&0xff);
  }

  double f = (156.25 * double(hs_div * (n1+1))) * double(1<<28)/ double(rfreq);

  logging::info("Read: hs_div %x  n1 %x  rfreq %lx  f %f MHz\n",
                hs_div, n1, rfreq, f);

  return f;
}

void Si570::program(int index)
{
  static const unsigned _hsd_div[] = { 7, 3 };
  static const unsigned _n1     [] = { 3, 3 };
  static const double   _rfreq  [] = { 5236., 5200. };
  reset();

  double fcal = read();

  //  Program for 1300/7 MHz

  //  Freeze DCO
  unsigned v = _reg[137];
  v |= (1<<4);
  _reg[137] = v;

  unsigned hs_div = _hsd_div[index];
  unsigned n1     = _n1     [index];
  uint64_t rfreq  = uint64_t(_rfreq[index] / fcal * double(1<<28));

  _reg[7] = ((hs_div&7)<<5) | ((n1>>2)&0x1f);
  _reg[8] = ((n1&3)    <<6) | ((rfreq>>32)&0x3f);
  _reg[9] = (rfreq>>24)&0xff;
  _reg[10]= (rfreq>>16)&0xff;
  _reg[11]= (rfreq>>8)&0xff;
  _reg[12]= (rfreq>>0)&0xff;
  
  logging::info("Wrote: hs_div %x  n1 %x  rfreq %lx  f %f MHz\n",
                hs_div, n1, rfreq, fcal);

  //  Unfreeze DCO
  v = _reg[137];
  v &= ~(1<<4);
  _reg[137] = v;

  v = _reg[135];
  v |= (1<<6);
  _reg[135] = v;

  read();
}
