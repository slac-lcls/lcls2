
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>
#include <arpa/inet.h>
#include <poll.h>
#include <signal.h>
#include <new>

#include "psdaq/hsd/Module64.hh"
#include "psdaq/hsd/Globals.hh"
#include "psdaq/hsd/TprCore.hh"
#include "psdaq/hsd/QABase.hh"
#include "psdaq/hsd/ClkSynth.hh"
#include "psdaq/hsd/Fmc134Cpld.hh"
#include "psdaq/hsd/Fmc134Ctrl.hh"
#include "psdaq/mmhw/AxiVersion.hh"
#include "psdaq/mmhw/RingBuffer.hh"

using Pds::Mmhw::AxiVersion;
using Pds::Mmhw::RingBuffer;

#include <string>

extern int optind;

using namespace Pds::HSD;

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -d <dev id>\n");
}

int main(int argc, char** argv) {

  extern char* optarg;
  char* endptr;

  const char* devName = "/dev/qadca";
  int c;
  bool lUsage = false;
  bool lClock = false;

  while ( (c=getopt( argc, argv, "Cd:")) != EOF ) {
    switch(c) {
    case 'd':
      devName = optarg;
      break;
    case 'C':
      lClock = true;
      break;
    case '?':
    default:
      lUsage = true;
      break;
    }
  }

  if (lUsage) {
    usage(argv[0]);
    exit(1);
  }

  int fd = open(devName, O_RDWR);
  if (fd<0) {
    perror("Open device failed");
    return -1;
  }

  Module64* p = Module64::create(fd);

  p->dumpMap();

  { AxiVersion& v = *reinterpret_cast<AxiVersion*>((char*)p->reg());
    printf("Axi Version [%p]: BuildStamp[%p]: %s\n", 
           &v, &(v.BuildStamp[0]), v.buildStamp().c_str()); }

  if (lClock) {
    uint32_t* monclk   =  reinterpret_cast<uint32_t*>((char*)p->reg()+0x98000);
    printf("%10.10s %10.10s L F S\n", "ClockName", "ClockRate");
    static const char* clockName[] = { "PllClk", "RxClk", "SysRef", "DevClk", "GtClk", "PgpClk", "TimClk", NULL };
    for(unsigned i=0; i<7; i++) {
      unsigned v = monclk[i+2];
      printf("%10.10s %10.3f %c %c %c\n",
             clockName[i], double(v&0x1fffffff)*1.e-6, 
             (v&(1<<31)) ? 'Y' : 'N',
             (v&(1<<30)) ? 'Y' : 'N',
             (v&(1<<29)) ? 'Y' : 'N' );
    }
      
    ClkSynth& clksynth = *reinterpret_cast<ClkSynth*>((char*)p->reg()+0x10400);
    TprCore&  tpr      = *reinterpret_cast<TprCore* >((char*)p->reg()+0x40000);

    //  Measure the clock rates (before)
    { 
      timespec tvb;
      clock_gettime(CLOCK_REALTIME,&tvb);
      unsigned vvb = tpr.TxRefClks;

      usleep(10000);

      timespec tve;
      clock_gettime(CLOCK_REALTIME,&tve);
      unsigned vve = tpr.TxRefClks;
      double dt = double(tve.tv_sec-tvb.tv_sec)+
        1.e-9*(double(tve.tv_nsec)-double(tvb.tv_nsec));
      double txclkr = 16.e-6*double(vve-vvb)/dt;
      printf("TxRefClk: %f MHz\n", txclkr);
    }

    p->i2c_lock(I2cSwitch::LocalBus);
    clksynth.setup(M3_7);
    clksynth.dump ();
    p->i2c_unlock();

    //
    //  Need a PLL reset on both tx and rx
    //
    p->tpr ().resetRxPll();
    p->base().resetFbPLL();
    usleep(10000);

    //  Measure the clock rates (after)
    printf("%10.10s %10.10s L F S\n", "ClockName", "ClockRate");
    for(unsigned i=0; i<7; i++) {
      unsigned v = monclk[i+2];
      printf("%10.10s %10.3f %c %c %c\n",
             clockName[i], double(v&0x1fffffff)*1.e-6, 
             (v&(1<<31)) ? 'Y' : 'N',
             (v&(1<<30)) ? 'Y' : 'N',
             (v&(1<<29)) ? 'Y' : 'N' );
    }
    { 
      timespec tvb;
      clock_gettime(CLOCK_REALTIME,&tvb);
      unsigned vvb = tpr.TxRefClks;

      usleep(10000);

      timespec tve;
      clock_gettime(CLOCK_REALTIME,&tve);
      unsigned vve = tpr.TxRefClks;
      double dt = double(tve.tv_sec-tvb.tv_sec)+
        1.e-9*(double(tve.tv_nsec)-double(tvb.tv_nsec));
      double txclkr = 16.e-6*double(vve-vvb)/dt;
      printf("TxRefClk: %f MHz\n", txclkr);
    }
    return 0;
  }

  p->i2c_lock(I2cSwitch::PrimaryFmc);

  //  Run I2C tests
  Fmc134Cpld* cpld = reinterpret_cast<Fmc134Cpld*>((char*)p->reg()+0x12800);
  // cpld->lmk_dump();
  // cpld->lmx_dump();
  // cpld->adc_dump(0);
  // cpld->adc_dump(1);

  //  Try a real configuration
  cpld->default_clocktree_init();

  cpld->default_adc_init();
  
  Fmc134Ctrl* ctrl = reinterpret_cast<Fmc134Ctrl*>((char*)p->reg()+0x81000);
  ctrl->default_init(*cpld,4);

  p->i2c_unlock();

  p->board_status();

  p->fmc_dump();

  return 0;
}

