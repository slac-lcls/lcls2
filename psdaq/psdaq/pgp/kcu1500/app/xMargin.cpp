
#include <sys/types.h>
#include <sys/mman.h>

#include <linux/types.h>
#include <unistd.h>
#include <stdio.h>
#include <termios.h>
#include <fcntl.h>
#include <sstream>
#include <string>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#include "DataDriver.h"

#include "GthEyeScan.hh"

using namespace Kcu;


static GthEyeScan* gth[8];
static const char* outfile = "eyescan.dat";
static unsigned prescale = 0;
static bool lsparse = false;

void* scan_routine(void* arg)
{
  unsigned lane = *(unsigned*)arg;
  printf("Start lane %u\n",lane);

  char ofile[64];
  sprintf(ofile,"%s.%u",outfile,lane);

  printf("gth[%u] @ %p\n",lane,gth[lane]);

  gth[lane]->enable(true);
  if (gth[lane]->enabled()) 
    gth[lane]->scan(ofile, prescale, 0, lsparse, true);
  else
    printf("enable failed\n");

  return 0;
}

void showUsage(const char* p) {
  printf("Usage: %s [options]\n", p);
  printf("Options:\n"
         "\t-P <dev>      Use pgpcard <dev> (integer)\n"
         "\t-L <lanes>    Bit mask of lanes\n"
         "\t-f <filename> Output file\n"
         "\t-p <prescale> Prescale exponent\n"
         "\t-s            Sparse mode\n");
}

using std::cout;
using std::endl;
using std::dec;
using std::hex;

int main (int argc, char **argv) {
  int           fd;
  unsigned      idev=0;
  unsigned      lanes=1;
  char dev[64];

  int c;

  while((c=getopt(argc,argv,"P:L:f:p:s")) != EOF) {
    switch(c) {
    case 'P': idev   = strtoul(optarg,NULL,0); break;
    case 'L': lanes  = strtoul(optarg,NULL,0); break;
    case 'f': outfile = optarg; break;
    case 'p': prescale = strtoul(optarg,NULL,0); break;
    case 's': lsparse = true; break;
    default:
      showUsage(argv[0]); return 0;
    }
  }

  sprintf(dev,"/dev/datadev_%u",idev);
  if ( (fd = open(dev, O_RDWR)) <= 0 ) {
    cout << "Error opening " << dev << endl;
    perror(dev);
    return(1);
  }

  Reg::set(fd);

  for(unsigned i=0; i<8; i++)
    gth[i] = new ((void*)(0x00809000+i*0x10000))GthEyeScan;

  pthread_t tid[8];
  unsigned lane[8];

  for(unsigned i=0; i<8 ;i++) {
    if (lanes & (1<<i)) {
      pthread_attr_t tattr;
      pthread_attr_init(&tattr);
      lane[i] = i;
      if (pthread_create(&tid[i], &tattr, &scan_routine, &lane[i]))
        perror("Error creating scan thread");
    }
  } 

  void* retval;
  for(unsigned i=0; i<8; i++)
    if (lanes & (1<<i))
      pthread_join(tid[i], &retval);

  return 0;
}
