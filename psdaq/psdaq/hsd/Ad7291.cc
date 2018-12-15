#include "psdaq/hsd/Ad7291.hh"
#include <stdio.h>
#include <unistd.h>
#include <string.h>

using namespace Pds::HSD;

#define REG(r) (_reg[r]&0xff)

void     Ad7291::start()
{
}

unsigned Ad7291::_read(unsigned ch)
{
  unsigned r = 0x20;
  if (ch<8) r |= (0x8000>>ch);
  else      r |= 0x80;
  _reg[0] = r;
  usleep(10000);

  unsigned v,a;
  do {
    if (ch<8)
      v = _reg[1];
    else
      v = _reg[2];
    a = (v>>12)&0xf;
    if (a == ch) break;
  } while(0);
  return v&0xfff;
}

Ad7291_Mon Ad7291::mon() 
{
  _reg[0] = 0x0022; // reset
  usleep(10000);

  Ad7291_Mon m;
  m.Tint = double(_read(8));

  for(unsigned i=0; i<8; i++)
    m.ain[i] = double(_read(i));

  return m;
}

#define DUMP(s) printf("%14.14s : %f\n",#s,s())

void FmcAdcMon::dump() const
{
  DUMP(adc0_1p1v_ana);
  DUMP(adc0_1p1v_dig);
  DUMP(adc0_1p9v_ana);
  DUMP(adc1_1p1v_ana);
  DUMP(adc1_1p1v_dig);
  DUMP(adc1_1p9v_ana);
  DUMP(adc0_temp);
  DUMP(adc1_temp);
  DUMP(int_temp);
}

void FmcVMon::dump() const
{
  DUMP(v5p5_osc100 );
  DUMP(v3p3_clock  );
  DUMP(v3p3_lmxpll );
  DUMP(vp_cpld_1p8v);
  DUMP(vadj        );
  DUMP(fmc_3p3v    );
  DUMP(fmc_12p0v   );
  DUMP(vio_m2c     );
  DUMP(int_temp    );
}

#undef DUMP
