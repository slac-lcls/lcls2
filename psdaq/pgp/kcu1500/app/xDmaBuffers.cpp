
#include <sys/types.h>
#include <sys/ioctl.h>
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

#include "DataDriver.h"

using namespace std;

void showUsage(const char* p) {
  printf("Usage: %s [options]\n", p);
  printf("Options:\n"
         "\t-n <buffers>\n"
         "\t-P <device>\n");
}

int main (int argc, char **argv) {

  int           fd;
  unsigned     nbuff = 2048;
  const char*  dev = "/dev/datadev_0";

  int c;

  while((c=getopt(argc,argv,"P:n:")) != EOF) {
    switch(c) {
    case 'P': dev    = optarg; break;
    case 'n': nbuff  = strtoul(optarg,NULL,0); break;
    default:
      showUsage(argv[0]);
      return 0;
    }
  }

  if ( (fd = open(dev, O_RDWR)) <= 0 ) {
    cout << "Error opening " << dev << endl;
    return(1);
  }

  printf("-- AxiStreamDmaV2Desc Buffers --\n");
  for(unsigned i=0; i<nbuff; i++) {
    uint32_t buff_lo, buff_hi;
    dmaReadRegister(fd, 0x4000+8*i, &buff_lo);
    dmaReadRegister(fd, 0x4004+8*i, &buff_hi);
    uint64_t buff = buff_hi; 
    buff <<= 32;
    buff |= buff_lo;
    printf("%016llx%c", buff, (i&7)==7 ? '\n':' ');
  }
    
  close(fd);
}
