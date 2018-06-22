
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
#include "psdaq/hsd/FexCfg.hh"
#include "psdaq/mmhw/AxiVersion.hh"

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
  printf("\t-E <exit>\n");
  printf("\t-C <close>\n");
  printf("\t-R <read>\n");
  printf("\t-W <write>\n");
  printf("\t-V <version>\n");
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
  bool lBufferStatus = false;
  bool lClkSync = false;
  bool lAdcSync = false;
  bool lRead = false;
  bool lWrite = false;
  bool lClose = false;
  bool lExit = false;
  bool lVersion = false;
  int  nPhase  = 0;
  
  while ( (c=getopt( argc, argv, "BCERVWSUF:")) != EOF ) {
    switch(c) {
    case 'B':
      lBufferStatus = true;
      break;
    case 'U':
      lClkSync = true;
      break;
    case 'C':
      lClose = true;
      break;
    case 'E':
      lExit = true;
      break;
    case 'R':
      lRead = true;
      break;
    case 'V':
      lVersion = true;
      break;
    case 'W':
      lWrite = true;
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

  if (lExit)
    return 0;

  if (lClose) {
    close(fd);
    return 0;
  }

  if (lWrite) {
    char buff[32];
    ::write(fd, buff, 1);
    close(fd);
    return 0;
  }

  if (lRead) {
    char buff[32];
    ::read(fd, buff, 1);
    close(fd);
    return 0;
  }

  Module* p = Module::create(fd);

  while (lVersion) {
    printf("BuildStamp: %s\n", p->version().buildStamp().c_str());
  }

  if (lAdcSync) {
    p->sync();
  }

  if (lClkSync) {
    p->clocktree_sync();
  }

  if (lBufferStatus) {
    p->dumpBase();
    return 0;
    FexCfg* fex = p->fex();
    unsigned v0 = fex[0]._base[0]._free;
    unsigned v1 = fex[0]._base[1]._free;
    printf("bufferStatus: %04x.%04x.%04x.%04x\n",
           (v0>> 0)&0xffff,
           (v0>>16)&0x1f,
           (v1>> 0)&0xffff,
           (v1>>16)&0x1f );
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
