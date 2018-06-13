
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>
#include <arpa/inet.h>
#include <signal.h>
#include <new>

#include "psdaq/hsd/Module.hh"
#include "psdaq/hsd/Globals.hh"

#include <string>

extern int optind;

using namespace Pds::HSD;

static double calc_phase(unsigned even,
                         unsigned odd)
{
  const double periodr = 1000./156.25;
  const double periodt = 7000./1300.;
  if (even)
    return double(even)/double(0x80000)*periodt;
  else
    return double(odd )/double(0x80000)*periodt + 0.5*periodr;
}

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -d <dev id>\n");
  printf("\t-S <sync ADC>\n");
  printf("\t-U <sync clocktree>\n");
  printf("\t-F <measure phases>\n");
}

int main(int argc, char** argv) {

  extern char* optarg;
  char* endptr;

  char qadc='a';
  int c;
  bool lUsage = false;
  bool lClkSync = false;
  bool lAdcSync = false;
  int  nPhase  = 0;

  while ( (c=getopt( argc, argv, "SUF:")) != EOF ) {
    switch(c) {
    case 'U':
      lClkSync = true;
      break;
    case 'S':
      lAdcSync = true;
      break;
    case 'F':
      nPhase = strtoul(optarg,NULL,0);
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

  char devname[16];
  sprintf(devname,"/dev/qadc%c",qadc);
  int fd = open(devname, O_RDWR);
  if (fd<0) {
    perror("Open device failed");
    return -1;
  }

  Module* p = Module::create(fd);

  if (lAdcSync) {
    p->sync();
  }

  if (lClkSync) {
    p->clocktree_sync();
  }

  while (nPhase) {
    unsigned trg_even  = p->trgPhase()[0];
    unsigned trg_odd   = p->trgPhase()[1];
    unsigned clkt_even = p->trgPhase()[2];
    unsigned clkt_odd  = p->trgPhase()[3];
    unsigned trgc_even = p->trgPhase()[4];
    unsigned trgc_odd  = p->trgPhase()[5];
    unsigned clkc_even = p->trgPhase()[6];
    unsigned clkc_odd  = p->trgPhase()[7];
    double trg_ph  = calc_phase(trg_even,trg_odd);
    double clkt_ph = calc_phase(clkt_even,clkt_odd);
    printf("Trigger Phase: %x/%x %x/%x %x/%x %x/%x [%f %f]\n",
           trg_even, trg_odd,
           clkt_even, clkt_odd,
           trgc_even, trgc_odd,
           clkc_even, clkc_odd,
           trg_ph, clkt_ph);
    if (--nPhase) {
      usleep(50000);
    }
  }

  return 0;
}
