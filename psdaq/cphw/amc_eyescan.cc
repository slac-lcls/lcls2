/**
 ** pgpdaq
 **
 **   Manage XPM and DTI to trigger and readout pgpcard (dev03)
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
#include "psdaq/cphw/Reg.hh"
#include "psdaq/cphw/GthEyeScan.hh"
#include "psdaq/cphw/AmcTiming.hh"
using Pds::Cphw::Reg;
using Pds::Cphw::GthEyeScan;
using Pds::Cphw::AmcTiming;

extern int optind;

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -a <IP addr (dotted notation)> : Use network <IP>\n");
  printf("         -f <filename>                  : Data output file\n");
  printf("         -s                             : Sparse scan\n");
  printf("         -p <prescale>                  : Sample count prescale (2**(17+<prescale>))\n");
}

static void* progress(void*)
{
  unsigned row, col;
  while(1) {
    sleep(60);
    Pds::Cphw::GthEyeScan::progress(row,col);
    printf("progress: %d,%d\n", row, col);
  }
  return 0;
}

int main(int argc, char** argv) {

  extern char* optarg;

  int c;

  const char* ip  = "10.0.1.103";
  unsigned prescale = 0;
  const char* outfile = "eyescan.dat";
  bool lsparse = false;

  while ( (c=getopt( argc, argv, "a:f:p:sh")) != EOF ) {
    switch(c) {
    case 'a': ip = optarg; break;
    case 'f': outfile = optarg; break;
    case 'p': prescale = strtoul(optarg,NULL,0); break;
    case 's': lsparse = true; break;
    case 'h': default:  usage(argv[0]); return 0;
    }
  }

  pthread_attr_t tattr;
  pthread_attr_init(&tattr);
  pthread_t tid;
  if (pthread_create(&tid, &tattr, &progress, 0))
    perror("Error creating progress thread");

  //  Setup AMC
  Pds::Cphw::Reg::set(ip, 8192, 0);
  GthEyeScan* gth = new ((void*)0x08c00000)GthEyeScan;
  if (!gth->enabled()) {
    gth->enable(true);
    // reset rx
    AmcTiming& amc = *new(0) AmcTiming;
    unsigned csr = amc.CSR;
    csr |= (1<<3);
    amc.CSR = csr;
    sleep(1);
  }
  gth->scan(outfile,prescale,1,lsparse);

  return 0;
}
