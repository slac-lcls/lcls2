
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
#include "psdaq/hsd/Fmc134Cpld.hh"
#include "psdaq/hsd/Fmc134Ctrl.hh"
#include "psdaq/mmhw/RingBuffer.hh"

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

  while ( (c=getopt( argc, argv, "d:")) != EOF ) {
    switch(c) {
    case 'd':
      devName = optarg;
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
  ctrl->default_init(*cpld);

  p->i2c_unlock();

  p->board_status();

  p->fmc_dump();

  return 0;
}

