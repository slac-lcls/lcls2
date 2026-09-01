#include "psdaq/hsd/src/OptFmc.hh"

#include <stdio.h>
#include <unistd.h>

using namespace Pds::HSD;

void OptFmc::resetPgp()
{
  unsigned v = this->qsfp;
  this->qsfp = v | (1<<4);  // assert qsfp reset
  usleep(10);
  this->qsfp = v;           // deassert qsfp reset
  usleep(1000);
  this->qsfp = v | (1<<5);  // assert tx reset
  usleep(10);
  this->qsfp = v;           // deassert tx reset
  usleep(1000);
  this->qsfp = v | (1<<6);  // assert rx reset
  usleep(10);
  this->qsfp = v;           // deassert rx reset
  usleep(1000);
}

int OptFmc::adcPhase()
{
    unsigned v = unsigned(phaseShift)>>16;
    if (v&(1<<15))
        return int(v - 0x10000);
    else
        return int(v);
}

void OptFmc::shiftAdcPhase(int tgt)
{
    int phase = adcPhase();
    int n = tgt-phase;

    unsigned v = 1;
    if (n>0) v = 3;

    while(n) {
        if (n>0) {
            n--;
        }
        else {
            n++;
        }
        phaseShift = v;
        usleep(1);
        printf("Wrote 0x%04x to phaseShift\n",v);
    }
}
