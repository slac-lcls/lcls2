
#include <sys/types.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
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
#include <linux/types.h>

#include "PgpDaq.hh"

using namespace std;

static void usage(const char* p)
{
  printf("Usage: %p <options>\n",p);
  printf("Options:\n");
  printf("\t-L [apps]          apps for each lane (CSV)\n");
  printf("\t-A <0/1>           autofill app 0\n");
  printf("\t-P <threshold>     set pause threshold\n");
  printf("\t-R                 reset app 0\n");
  printf("\t-d <n>             add descriptors\n");
}

int main (int argc, char **argv) {

  int          fd;
  const char*  dev = "/dev/pgpdaq0";
  bool         lReset   = false;
  int          autoFill = -1;
  unsigned     apps[LANES];
  int          pauseTh  = -1;
  //unsigned     ndescriptors = 0;
  int c;

  memset(apps, 0, sizeof(apps));

  while((c=getopt(argc,argv,"A:mMd:L:P:s:r:R")) != EOF) {
    switch(c) {
    case 'A': autoFill = strtoul(optarg, NULL, 0); break;
    case 'R': lReset   = true; break;
    case 'L': 
      { char* endptr = optarg;
        for(unsigned i=0; i<LANES; i++) {
          apps[i] = strtoul(endptr, &endptr, 0);
          endptr++;
        } } break;
    case 'P': pauseTh        = strtoul(optarg, NULL, 0); break;
    //case 'd': ndescriptors   = strtoul(optarg, NULL, 0); break;
    default: usage(argv[0]); return 0;
    }
  }

  if ( (fd = open(dev, O_RDWR)) <= 0 ) {
    cout << "Error opening " << dev << endl;
    return(1);
  }

  PgpDaq::PgpCard* p = (PgpDaq::PgpCard*)mmap(NULL, 0x01000000, (PROT_READ|PROT_WRITE), (MAP_SHARED|MAP_LOCKED), fd, 0);   

  if (lReset)
    p->reset = 1;

  if (autoFill > 0)
    p->clients[0].autoFill = 1;
  else if (autoFill == 0)
    p->clients[0].autoFill = 0;

  if (pauseTh >= 0)
    for(unsigned i=0; i<LANES; i++)
      p->dmaLane[i].blocksPause = pauseTh;

  for(unsigned i=0; i<LANES; i++)
    p->dmaLane[i].client = apps[i];

  close(fd);
  return 0;
}
