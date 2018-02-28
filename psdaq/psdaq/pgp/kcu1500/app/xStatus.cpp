
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

  int          fd;
  const char*  dev = "/dev/datadev_0";
  unsigned     base;

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

  { AxiVersion vsn;
    if (axiVersionGet(fd, &vsn)>=0) {
      printf("-- Core Axi Version --\n");
      printf("firmwareVersion : %x\n", vsn.firmwareVersion);
      printf("upTimeCount     : %u\n", vsn.upTimeCount);
      printf("deviceId        : %x\n", vsn.deviceId);
      printf("buildString     : %s\n", vsn.buildString); 
    }
  }

  { AxiVersion vsn;
    dmaReadRegister(fd, 0x008a0000, &vsn.firmwareVersion);
    dmaReadRegister(fd, 0x008a0008, &vsn.upTimeCount);
    dmaReadRegister(fd, 0x008a0500, &vsn.deviceId);
    for(unsigned i=0; i<64; i++)
      dmaReadRegister(fd, 0x008a0800+4*i, reinterpret_cast<uint32_t*>(&vsn.buildString[i*4]));

    printf("-- Appl Axi Version --\n");
    printf("firmwareVersion : %x\n", vsn.firmwareVersion);
    printf("upTimeCount     : %u\n", vsn.upTimeCount);
    printf("deviceId        : %x\n", vsn.deviceId);
    printf("buildString     : %s\n", vsn.buildString); 
  }

#define READREG(name,addr)                   \
    if (dmaReadRegister(fd, addr, &reg)<0) { \
      perror(#name);                         \
      return -1;                             \
    }
#define PRINTFIELD(name, addr, offset, mask) {                  \
    uint32_t reg;                                               \
    printf("%20.20s :", #name);                                 \
    for(unsigned i=0; i<8; i++) {                               \
      READREG(name,addr+base+i*0x10000);                        \
      printf(" %8x", (reg>>offset)&mask);                       \
    }                                                           \
    printf("\n"); }
#define PRINTBIT(name, addr, bit)  PRINTFIELD(name, addr, bit, 1)
#define PRINTREG(name, addr)       PRINTFIELD(name, addr,   0, 0xffffffff)
#define PRINTERR(name, addr)       PRINTFIELD(name, addr,   0, 0xf)
#define PRINTFRQ(name, addr) {                                  \
    uint32_t reg;                                               \
    printf("%20.20s :", #name);                                 \
    for(unsigned i=0; i<8; i++) {                               \
      dmaReadRegister(fd, addr+0x00808000+i*0x10000, &reg);     \
      printf(" %8.3f", float(reg)*1.e-6);                       \
    }                                                           \
    printf("\n"); }

  printf("-- PgpAxiL Registers --\n");
  base = 0x00808000;
  PRINTFIELD(loopback , 0x08, 0, 0x7);
  PRINTBIT(phyRxActive, 0x10, 0);
  PRINTBIT(locLinkRdy , 0x10, 1);
  PRINTBIT(remLinkRdy , 0x10, 2);
  PRINTERR(cellErrCnt , 0x14);
  PRINTERR(linkDownCnt, 0x18);
  PRINTERR(linkErrCnt , 0x1c);
  PRINTFIELD(remRxOflow , 0x20,  0, 0xffff);
  PRINTFIELD(remRxPause , 0x20, 16, 0xffff);
  PRINTREG(rxFrameCnt , 0x24);
  PRINTERR(rxFrameErrCnt, 0x28);
  PRINTFRQ(rxClkFreq  , 0x2c);
  PRINTERR(rxOpCodeCnt, 0x30);
  PRINTREG(rxOpCodeLst, 0x34);
  PRINTERR(phyRxIniCnt, 0x130);

  PRINTBIT(flowCntlDis, 0x80, 0);
  PRINTBIT(txDisable  , 0x80, 1);
  PRINTBIT(phyTxActive, 0x84, 0);
  PRINTBIT(linkRdy    , 0x84, 1);
  PRINTFIELD(locOflow   , 0x8c, 0,  0xffff);
  PRINTFIELD(locPause   , 0x8c, 16, 0xffff);
  PRINTREG(txFrameCnt , 0x90);
  PRINTERR(txFrameErrCnt, 0x94);
  PRINTFRQ(txClkFreq  , 0x9c);
  PRINTERR(txOpCodeCnt, 0xa0);
  PRINTREG(txOpCodeLst, 0xa4);

  printf("-- AppTxSim Registers --\n");
  base = 0x00900000;

  { uint32_t reg;
    READREG(control ,0x00900100);
    printf("%20.20s : %8x\n","control",reg);

    READREG(size    ,0x00900104);
    printf("%20.20s : %8x\n","size",reg);
    
    READREG(overflow,0x00900000);
    printf("%20.20s :","overflow");
    for(unsigned i=0; i<8; i++)
      printf(" %8x", (reg>>(4*i))&0xf);
    printf("\n");
  }

  close(fd);
}
