
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
         "\t-P <dev>   Use device <dev> (integer)\n"
         "\t-L <lanes> Bit mask of loopback lanes\n");
}

int main (int argc, char **argv) {

  int           fd;
  unsigned      lanes=1;
  const char*   dev = "/dev/datadev_0";
  int c;

  while((c=getopt(argc,argv,"P:L:")) != EOF) {
    switch(c) {
    case 'P': dev    = optarg; break;
    case 'L': lanes  = strtoul(optarg,NULL,0); break;
    default:
      showUsage(argv[0]); return 0;
    }
  }

  if ( (fd = open(dev, O_RDWR)) <= 0 ) {
    cout << "Error opening " << dev << endl;
    return(1);
  }

  for(unsigned i=0; i<8; i++) {
    unsigned v;
    dmaReadRegister(fd, 0x00808008+i*0x10000, &v);
    v &= 0x7;
    if (lanes & (1<<i))
      v |= 2;
    dmaWriteRegister(fd, 0x00808008+i*0x10000, v);
  }

  close(fd);
  return 0;
}
