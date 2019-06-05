
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

static void usage(const char* p)
{
  printf("Usage: %s [options]\n",p);
  printf("Options:\n");
  printf("-d <device>\n");
  printf("-F <fifoThreshold>\n");
}

int main (int argc, char **argv) {

  int           fd;
  const char*   dev = "/dev/datadev_0";
  int           fifoThr = -1;
  int           c;

  while((c=getopt(argc,argv,"d:F:"))!=-1) {
    switch(c) {
    case 'd': dev = optarg; break;
    case 'F': fifoThr = strtoul(optarg, NULL, 0); break;
    default:  usage(argv[0]); return 1;
    }
  }


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
  PRINTREG(fifoOF  , 0xa0);

  if (fifoThr > 0)
    dmaWriteRegister(fd, base+0xa4, fifoThr);

  PRINTREG(fifoTh  , 0xa4);
  PRINTREG(fifoDep , 0xa8);
  PRINTREG(fifoDep , 0xac);
  
  PRINTCLK(axilOther  ,0x100);
  PRINTCLK(timingRef  ,0x104);
  PRINTCLK(migA       ,0x108);
  PRINTCLK(migB       ,0x10c);

  close(fd);
}
