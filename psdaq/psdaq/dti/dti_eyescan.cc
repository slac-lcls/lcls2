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
#include "psdaq/cphw/GthEyeScan.hh"
using Pds::Cphw::GthEyeScan;

extern int optind;

static GthEyeScan* gth;
static const char* outfile = "eyescan.dat";
static unsigned prescale = 0;
static bool lsparse = false;
static bool lhscan = true;

void* scan_routine(void* arg)
{
  unsigned lane = *(unsigned*)arg;
  printf("Start lane %u\n",lane);

  char ofile[64];
  sprintf(ofile,"%s.%u",outfile,lane);

  gth[lane].enable(true);
  gth[lane].scan(ofile, prescale, 0, lsparse, lhscan);

  return 0;
}

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -a <IP addr (dotted notation)> : Use network <IP>\n");
  printf("         -f <filename>                  : Data output file\n");
  printf("         -s                             : Sparse scan\n");
  printf("         -v                             : Vertical scan\n");
  printf("         -p <prescale>                  : Sample count prescale (2**(17+<prescale>))\n");
  printf("         -l <lane mask>                 : Bit maks of lanes to scan (default=1)\n");
}

int main(int argc, char** argv) {

  extern char* optarg;
  int c;

  const char* ip  = "10.0.1.103";
  unsigned lanes = 1;

  while ( (c=getopt( argc, argv, "a:f:l:p:svh")) != EOF ) {
    switch(c) {
    case 'a': ip = optarg; break;
    case 'f': outfile = optarg; break;
    case 'p': prescale = strtoul(optarg,NULL,0); break;
    case 's': lsparse = true; break;
    case 'v': lhscan = false; break;
    case 'l': lanes = strtoul(optarg,NULL,0); break;
    case 'h': default:  usage(argv[0]); return 0;
    }
  }

  //  Setup DTI
  Pds::Cphw::Reg::set(ip, 8192, 0);

  gth = new ((void*)0xb0000000)GthEyeScan;

  pthread_t tid[7];
  unsigned lane[7];

  for(unsigned i=0; i<7 ;i++) {
    if (lanes & (1<<i)) {
      pthread_attr_t tattr;
      pthread_attr_init(&tattr);
      lane[i] = i;
      if (pthread_create(&tid[i], &tattr, &scan_routine, &lane[i]))
        perror("Error creating scan thread");
    }
  } 

  void* retval;
  for(unsigned i=0; i<7; i++)
    if (lanes & (1<<i))
      pthread_join(tid[i], &retval);

  return 0;
}
