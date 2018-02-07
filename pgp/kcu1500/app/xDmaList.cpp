
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

int main (int argc, char **argv) {

  int           fd;
  const char*  dev = "/dev/datadev_0";

  if (argc>2 || 
      (argc==2 && argv[1][0]=='-')) {
    printf("Usage: %s [<device>]\n", argv[0]);
    return(0);
  }
  else if (argc==2)
    dev = argv[1];

  if ( (fd = open(dev, O_RDWR)) <= 0 ) {
    cout << "Error opening " << dev << endl;
    return(1);
  }

  printf("-- AxiStreamDmaV2Desc Buffers --\n");
  for(unsigned i=0; i<2048; i++) {
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
