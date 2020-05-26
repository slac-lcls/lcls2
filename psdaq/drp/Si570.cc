#include "Si570.hh"
#include "DataDriver.h"

using namespace Drp;

Si570::Si570(int fd, unsigned off) : _fd(fd), _off(off) {}
Si570::~Si570() {}

void Si570::reset()
{
  unsigned v;
  dmaReadRegister(_fd, _off+4*135, &v);
  v |= 1;
  dmaWriteRegister(_fd, _off+4*135, v);
  do { 
    usleep(100); 
    dmaReadRegister(_fd, _off+4*135, &v); 
  } while (v&1);
}

double Si570::read()
{
  //  Read factory calibration for 156.25 MHz
  static const unsigned hsd_divn[] = {4,5,6,7,0,9,0,11};
  unsigned v;
  dmaReadRegister(_fd, _off+4*7, &v);
  printf("si570[7] = 0x%x\n", v);
  unsigned hs_div = hsd_divn[(v>>5)&7];
  unsigned n1 = (v&0x1f)<<2;
  dmaReadRegister(_fd, _off+4*8, &v); 
  printf("si570[8] = 0x%x\n", v);
  n1 |= (v>>6)&3;
  uint64_t rfreq = v&0x3f;
  for(unsigned i=9; i<13; i++) {
    dmaReadRegister(_fd, _off+4*i, &v);
    printf("si570[%d] = 0x%x\n", i, v);
    rfreq <<= 8;
    rfreq |= (v&0xff);
  }

  double f = (156.25 * double(hs_div * (n1+1))) * double(1<<28)/ double(rfreq);

  printf("Read: hs_div %x  n1 %x  rfreq %lx  f %f MHz\n",
         hs_div, n1, rfreq, f);

  return f;
}

void Si570::program()
{
  reset();

  double fcal = read();

  //  Program for 1300/7 MHz

  //  Freeze DCO
  unsigned v;
  dmaReadRegister(_fd, _off+4*137, &v);
  v |= (1<<4);
  dmaWriteRegister(_fd, _off+4*137, v);

  unsigned hs_div = 3; // =7
  unsigned n1     = 3; // =4
  uint64_t rfreq  = uint64_t(5200. / fcal * double(1<<28));

  dmaWriteRegister(_fd, _off+4*7 , ((hs_div&7)<<5) | ((n1>>2)&0x1f) );
  dmaWriteRegister(_fd, _off+4*8 , ((n1&3)    <<6) | ((rfreq>>32)&0x3f) );
  dmaWriteRegister(_fd, _off+4*9 , (rfreq>>24)&0xff );
  dmaWriteRegister(_fd, _off+4*10, (rfreq>>16)&0xff );
  dmaWriteRegister(_fd, _off+4*11, (rfreq>>8)&0xff );
  dmaWriteRegister(_fd, _off+4*12, (rfreq>>0)&0xff );
  
  printf("Wrote: hs_div %x  n1 %x  rfreq %lx  f %f MHz\n",
         hs_div, n1, rfreq, fcal);

  //  Unfreeze DCO
  dmaReadRegister(_fd, _off+4*137, &v);
  v &= ~(1<<4);
  dmaWriteRegister(_fd, _off+4*137, v);
  
  dmaReadRegister(_fd, _off+4*135, &v);
  v |= (1<<6);
  dmaWriteRegister(_fd, _off+4*135, v);

  read();
}
