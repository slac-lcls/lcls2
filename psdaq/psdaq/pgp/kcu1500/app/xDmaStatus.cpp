
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

#define PRINTFIELD(name, addr, offset, mask) {                  \
    uint32_t reg;                                               \
    printf("%20.20s :", #name);                                 \
    if (dmaReadRegister(fd, base+addr, &reg)<0) {               \
      perror(#name);                                            \
      return -1;                                                \
    }                                                           \
    printf(" 0x%8x\n", (reg>>offset)&mask); }
#define PRINTBIT(name, addr, bit)  PRINTFIELD(name, addr, bit, 1)
#define PRINTREG(name, addr)       PRINTFIELD(name, addr,   0, 0xffffffff)
#define PRINTCLK(name, addr) {                                  \
    uint32_t reg;                                               \
    printf("%20.20s :", #name);                                 \
    if (dmaReadRegister(fd, base+addr, &reg)<0) {               \
      perror(#name);                                            \
      return -1;                                                \
    }                                                           \
    printf(" %5.3f MHz\n", double(reg&0xfffffff)*1.e-6);        \
}
#define PRINTERR(name, addr)       PRINTFIELD(name, addr,   0, 0xf)

  printf("-- AxiStreamDmaV2Desc Registers --\n");
  unsigned base = 0x00800000;
  PRINTBIT(enable   , 0x00, 0);

  PRINTREG(blockSize   , 0x80);
  PRINTFIELD(blocksPause, 0x84, 8, 0xff);
  PRINTFIELD(blocksFree, 0x88, 0, 0xfff);
  PRINTFIELD(blocksQued, 0x88,12, 0xfff);
  PRINTREG(writeQueCnt , 0x8c);
  PRINTREG(wrndex  , 0x90);
  PRINTREG(wcIndex , 0x94);
  PRINTREG(rdIndex , 0x98);
  PRINTREG(fifoOF  , 0x9c);

  PRINTCLK(axilOther  ,0x100);
  PRINTCLK(timingRef  ,0x104);
  PRINTCLK(migA       ,0x108);
  PRINTCLK(migB       ,0x10c);

  close(fd);
}
