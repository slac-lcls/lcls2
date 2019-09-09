#include "psdaq/hsd/AdcSync.hh"

#include <stdio.h>
#include <unistd.h>

using namespace Pds::HSD;

void AdcSync::set_delay(const unsigned* delay)
{
  for(unsigned i=0; i<4; i++)
    _delay[2*i] = delay[i];
  
  unsigned v = _cmd;
  usleep(1);
  _cmd = v | (0xf<<16);
  usleep(100);
  _cmd = v;

  for(unsigned i=0; i<4; i++)
    printf("SyncBit: %u  DelayIn: %u  DelaySet: %u  DelayOut: %u\n",
           i, delay[i], _delay[2*i], _delay[2*i+1]);
}

void AdcSync::start_training()
{
  //  _cmd = v | 1;
  _cmd = (4095<<1) | 1;
  printf("AdcSync set %x\n",_cmd);
}

void AdcSync::stop_training()
{
  unsigned v = _cmd;
  _cmd = v & ~1;
  printf("AdcSync set %x\n",_cmd);
}

void AdcSync::dump_status() const
{
  printf("-- match v delay --\n");
  for(unsigned j=0; j<4; j++) {
    for(unsigned k=0; k<8; k++) {
      _select = j + ((7-k)<<4);
      unsigned v = _match;
      //      printf("%08x", _match);
      for(unsigned m=0; m<32; m++)
        printf("%c", v&(1<<(31-m)) ? '+':'.');
      //      printf("%08x", _match);
    }
    printf("\n");
  }
}
