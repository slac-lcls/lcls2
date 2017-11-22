#include "psdaq/cphw/AmcPLL.hh"
#include "psdaq/cphw/Utils.hh"

#include <stdio.h>
#include <unistd.h>

using namespace Pds::Cphw;

void AmcPLL::BwSel(unsigned sel)
{
  setf(_config,sel,4,0);
}
unsigned AmcPLL::BwSel() const
{
  return getf(_config,4,0);
}

void AmcPLL::FrqTbl(unsigned sel)
{
  setf(_config,sel,2,4);
}
unsigned AmcPLL::FrqTbl() const
{
  return getf(_config,2,4);
}

void AmcPLL::FrqSel(unsigned sel)
{
  setf(_config,sel,8,8);
}
unsigned AmcPLL::FrqSel() const
{
  return getf(_config,8,8);
}

void AmcPLL::RateSel(unsigned sel)
{
  setf(_config,sel,4,16);
}
unsigned AmcPLL::RateSel() const
{
  return getf(_config,4,16);
}

void AmcPLL::PhsInc()
{
  unsigned v = _config;
  _config = v|(1<<20);
  usleep(10);
  _config = v;
  usleep(10);
}

void AmcPLL::PhsDec()
{
  unsigned v = _config;
  _config = v|(1<<21);
  usleep(10);
  _config = v;
  usleep(10);
}

void AmcPLL::Bypass(bool v)
{
  setf(_config,v?1:0,1,22);
}
bool AmcPLL::Bypass() const
{
  return getf(_config,1,22);
}

void AmcPLL::Reset()
{
  setf(_config,0,1,23);
  usleep(10);
  setf(_config,1,1,23);
}

unsigned AmcPLL::Status0() const
{
  return getf(_config,1,27);
}

unsigned AmcPLL::Count0() const
{
  return getf(_config,3,24);
}

unsigned AmcPLL::Status1() const
{
  return getf(_config,1,31);
}

unsigned AmcPLL::Count1() const
{
  return getf(_config,3,28);
}

void AmcPLL::Skew(int skewSteps)
{
  unsigned v = _config;
  while(skewSteps > 0) {
    _config = v|(1<<20);
    usleep(10);
    _config = v;
    usleep(10);
    skewSteps--;
  }
  while(skewSteps < 0) {
    _config = v|(1<<21);
    usleep(10);
    _config = v;
    usleep(10);
    skewSteps++;
  }
}

void AmcPLL::dump() const
{
  unsigned bwSel = getf(_config,4,0);
  unsigned frTbl = getf(_config,2,4);
  unsigned frSel = getf(_config,8,8);
  unsigned rate  = getf(_config,4,16);
  unsigned cnt0  = getf(_config,3,24);
  unsigned stat0 = getf(_config,1,27);
  unsigned cnt1  = getf(_config,3,28);
  unsigned stat1 = getf(_config,1,31);

  static const char lmh[] = {'L', 'H', 'M', 'm'};
  printf("  ");
  printf("FrqTbl %c  ", lmh[frTbl]);
  printf("FrqSel %c%c%c%c  ", lmh[getf(frSel,2,6)], lmh[getf(frSel,2,4)],
                             lmh[getf(frSel,2,2)], lmh[getf(frSel,2,0)]);
  printf("BwSel %c%c  ", lmh[getf(bwSel,2,2)], lmh[getf(bwSel,2,0)]);
  printf("Rate %c%c  ", lmh[getf(rate,2,2)], lmh[getf(rate,2,0)]);
  printf("Cnt0 %u  ", cnt0);
  printf("LOS %c  ", stat0 ? 'Y' : 'N');
  printf("Cnt1 %u  ", cnt1);
  printf("LOL %c  ", stat1 ? 'Y' : 'N');
  printf("\n");
}
