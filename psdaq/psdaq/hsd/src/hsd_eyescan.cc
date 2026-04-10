/**
 ** pgpdaq
 **
 **/

#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <signal.h>
#include <new>
#include "psdaq/mmhw/AxiVersion.hh"
#include "psdaq/mmhw/GthEyeScan.hh"
using Pds::Mmhw::AxiVersion;
using Pds::Mmhw::GthEyeScan;

extern int optind;

static GthEyeScan* gth;
static const char* outfile = "eyescan.dat";
static unsigned prescale = 0;
static bool lsparse = false;

void* scan_routine(void* arg)
{
  unsigned lane = *(unsigned*)arg;

  char ofile[64];
  sprintf(ofile,"%s.%u",outfile,lane);

  gth[lane].scan(ofile, prescale, 0, lsparse);

  return 0;
}

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -d <dev>                       : Use device <dev>\n");
  printf("         -f <filename>                  : Data output file\n");
  printf("         -s                             : Sparse scan\n");
  printf("         -p <prescale>                  : Sample count prescale (2**(17+<prescale>))\n");
}

int main(int argc, char** argv) {

  extern char* optarg;
  int c;

  const char* dev = "/dev/qadca";

  while ( (c=getopt( argc, argv, "d:f:p:sh")) != EOF ) {
    switch(c) {
    case 'd': dev = optarg; break;
    case 'f': outfile = optarg; break;
    case 'p': prescale = strtoul(optarg,NULL,0); break;
    case 's': lsparse = true; break;
    case 'h': default:  usage(argv[0]); return 0;
    }
  }

  int fd = open(dev, O_RDWR);
  if (fd<0) {
    perror("Open device failed");
    return -1;
  }

  const unsigned nlane=4;
  size_t mapsz = 0x800*nlane + 0x91000;

  void* ptr = mmap(0, mapsz, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    perror("Failed to map");
    return 0;
  }

  printf("BuildStamp: %s\n", (new (ptr) AxiVersion)->buildStamp().c_str());

  gth = reinterpret_cast<GthEyeScan*>((char*)ptr+0x91000);

  pthread_t tid[nlane];
  unsigned lane[nlane];

  for(unsigned i=0; i<nlane ;i++) {
    pthread_attr_t tattr;
    pthread_attr_init(&tattr);
    lane[i] = i;
    if (pthread_create(&tid[i], &tattr, &scan_routine, &lane[i]))
      perror("Error creating scan thread");
  }

  void* retval;
  for(unsigned i=0; i<nlane; i++)
    pthread_join(tid[i], &retval);

  return 0;
}
