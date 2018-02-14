#include "psdaq/hsd/AdcCore.hh"

#include <unistd.h>
#include <stdio.h>
#include <string.h>

using namespace Pds::HSD;

void AdcCore::dump_status() const
{
  printf("cal_status before : %x\n", _status);
}

void AdcCore::init_training(unsigned ref_delay)
{
  //Reset clock buffer and iDelays first
  _cmd = 0x3;
  usleep(50000);

  _adrclk_delay_set_auto = ref_delay;
  _master_start  = 0x140;
}

void AdcCore::start_training()
{
  //Reset clock buffer and iDelays first
  //  _cmd = 0x3;
  //  usleep(50000);

  //  _adrclk_delay_set_auto = 0x08;
  usleep(500000);

  //then reset iSerdes, when the clocks are stable
  _cmd = 0x4;
  usleep(50000);

  //IO training is done on the FMC126

  //Start training
  //  Search for idelay window start 1/4 into idelay range (full range is 512)
  //  _master_start  = 0x140;

  _cmd = 0x08;
  usleep(50000);

  while(1) {
    unsigned v = _status;
    if ((v&0x3F)==0x36) {
      printf("FMC Ready\n");
      break;
    }
    else {
      printf("FMC Busy [%x]\n",v);
      usleep(100000);
      break;
    }
  }
}

void AdcCore::dump_training()
{
  //  Read the results
  printf("\tFMC\n");
  for(unsigned i=0; i<44; i++) {
    _channel_select = i;
    unsigned csel = _channel_select;
    asm volatile("lfence" ::: "memory");
    unsigned mhi = _tap_match_hi;
    unsigned mlo = _tap_match_lo;
    unsigned tap = _adc_req_tap;
    printf("[Ch%c:b%02u] 0x%08x%08x : %u (%f ps)\n",
           'A'+csel/11,csel%11,mhi,mlo,tap,double(tap)*1250./512.);
  }
}
  
void AdcCore::pulse_sync   ()
{
  _cmd   = (1<<4);
}

void AdcCore::loop_checking()
{
  int32_t CheckTime = 10;
  int32_t ch_fails[4]; memset(ch_fails,0,4*sizeof(int32_t));

  for(int i=0; i<CheckTime; i++) {
    //Reset pattern error flag
    volatile unsigned v = _status;
    //Test for one second
    usleep(100000);
    //Read pattern error flags
    v = _status;
    if (v&0x1000)      ch_fails[0]++;
    if (v&0x2000)      ch_fails[1]++;
    if (v&0x4000)      ch_fails[2]++;
    if (v&0x8000)      ch_fails[3]++;
    // Report
    printf("Pattern check   : %f seconds ", double(i+1)*0.1);
    for(unsigned i=0; i<4; i++)
      printf("| ADC%u: %d", i, ch_fails[i]);
    printf("\n");
  }
}
