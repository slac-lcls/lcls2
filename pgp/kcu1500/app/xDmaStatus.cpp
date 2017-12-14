
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
    if (dmaReadRegister(fd, addr, &reg)<0) {                    \
      perror(#name);                                            \
      return -1;                                                \
    }                                                           \
    printf(" %8x\n", (reg>>offset)&mask); }
#define PRINTBIT(name, addr, bit)  PRINTFIELD(name, addr, bit, 1)
#define PRINTREG(name, addr)       PRINTFIELD(name, addr,   0, 0xffffffff)
#define PRINTERR(name, addr)       PRINTFIELD(name, addr,   0, 0xf)

  printf("-- AxiStreamDmaV2Desc Registers --\n");
  PRINTBIT(enable   , 0x00, 0);
  PRINTFIELD(version, 0x00, 24, 0xff);
  PRINTBIT(intEnable, 0x04, 0);
  PRINTBIT(contEn   , 0x08, 0);
  PRINTBIT(dropEn   , 0x0c, 0);
  PRINTREG(wrBaseAddL , 0x10);
  PRINTREG(wrBaseAddH , 0x14);
  PRINTREG(rdBaseAddL , 0x18);
  PRINTREG(rdBaseAddH , 0x1c);
  PRINTREG(buBaseAddrH, 0x24);
  PRINTREG(maxSize    , 0x28);
  PRINTFIELD(online     , 0x2c, 0, 0xff);
  PRINTFIELD(acknowledge, 0x30, 0, 0xff);
  PRINTFIELD(chanCount  , 0x34, 0, 0xff);
  PRINTFIELD(descAWidth , 0x38, 0, 0xff);
  PRINTFIELD(descCache  , 0x3c, 0, 0xf);
  PRINTFIELD(buffCache  , 0x3c, 8, 0xf);
  PRINTREG(fifoDin      , 0x40);  
  PRINTFIELD(intAckCnt  , 0x4c, 0, 0xffff);
  PRINTREG(intReqCnt  , 0x50);
  PRINTREG(wrIndex    , 0x54);
  PRINTREG(rdIndex    , 0x58);
  PRINTREG(wrReqMiss  , 0x5c);

  close(fd);
}
