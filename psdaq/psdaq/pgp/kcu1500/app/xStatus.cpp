
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

static void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -l <mask>\n");
}

int main (int argc, char **argv) {

  int          fd  [2];
  unsigned     base;
  int          lbmask = -1;
  int          c;

  while((c=getopt(argc,argv,"l:"))!=-1) {
    switch(c) {
    case 'l': lbmask = strtoul(optarg,NULL,0); break;
    default:  usage(argv[0]); return 1;
    }
  }

  if ( (fd[0] = open("/dev/datadev_0", O_RDWR)) <= 0 ) {
    cout << "Error opening /dev/datadev_0" << endl;
    return(1);
  }

  if ( (fd[1] = open("/dev/datadev_1", O_RDWR)) <= 0 ) {
    cout << "Error opening /dev/datadev_1" << endl;
    return(1);
  }

  { AxiVersion vsn;
    if (axiVersionGet(fd[0], &vsn)>=0) {
      printf("-- Core Axi Version --\n");
      printf("firmwareVersion : %x\n", vsn.firmwareVersion);
      printf("upTimeCount     : %u\n", vsn.upTimeCount);
      printf("deviceId        : %x\n", vsn.deviceId);
      printf("buildString     : %s\n", vsn.buildString); 
    }
  }

#define READREG(name,addr)                   \
    if (dmaReadRegister(ifd, addr, &reg)<0) { \
      perror(#name);                         \
      return -1;                             \
    }
#define PRINTFIELD(name, addr, offset, mask) {                  \
    uint32_t reg;                                               \
    printf("%20.20s :", #name);                                 \
    for(unsigned i=0; i<8; i++) {                               \
      int ifd = fd[i>>2];                                       \
      READREG(name,addr+base+(i&3)*0x10000);                    \
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
      dmaReadRegister(fd[i>>2], addr+base+(i&3)*0x10000, &reg); \
      printf(" %8.3f", float(reg)*1.e-6);                       \
    }                                                           \
    printf("\n"); }

  base = 0x00a08000;

  if (lbmask != -1) {
    unsigned reg;
    for(unsigned i=0; i<8; i++) {
      int ifd = fd[i>>2];
      unsigned a = 0x08+base+(i&3)*0x10000;
      READREG(loopback, a);
      reg &= ~0x7;
      if (lbmask & (1<<i))
        reg |= 0x2;
      dmaWriteRegister(ifd,a,reg);
    }
  }

  printf("-- PgpAxiL Registers --\n");
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

  close(fd[0]);
  close(fd[1]);
}
