
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

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -d <dev id>\n");
  printf("\t-A [A]\n");
  printf("\t-B [B]\n");
  printf("\t-P [P]\n");
  printf("\t-R [R]\n");
  printf("\t-c [charge pump]\n");
  printf("\t-a [antibacklash]\n");
}

int main(int argc, char** argv) {

  extern char* optarg;
  char* endptr;

  char qadc='a';
  int c;
  bool lUsage = false;
  int A=2, B=21, P=6, R=1, cp=3, ab=0;

  while ( (c=getopt( argc, argv, "A:B:P:R:c:a:h?")) != EOF ) {
    switch(c) {
    case 'A': A  = strtoul(optarg,NULL,0); break;
    case 'B': B  = strtoul(optarg,NULL,0); break;
    case 'P': P  = strtoul(optarg,NULL,0); break;
    case 'R': R  = strtoul(optarg,NULL,0); break;
    case 'c': cp = strtoul(optarg,NULL,0); break;
    case 'a': ab = strtoul(optarg,NULL,0); break;
    default:
      lUsage=true;
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

  p->fmc_modify(A,B,P,R,cp,ab);
  p->train_io(0);

  return 0;
}
