
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
#include "PgpCam.hh"

using namespace std;

static void usage(const char* p)
{
  printf("Usage: %s <options>\n",p);
  printf("Options:\n");
  printf("\t-d <device>  [e.g. /dev/pgpcam0]\n");
}

int main (int argc, char **argv) {

  int          fd;
  const char*  dev = "/dev/pgpcam0";
  int c;

  while((c=getopt(argc,argv,"d:")) != EOF) {
    switch(c) {
    case 'd': dev    = optarg; break;
    default: usage(argv[0]); return 0;
    }
  }

  if ( (fd = open(dev, O_RDWR)) <= 0 ) {
    cout << "Error opening " << dev << endl;
    return(1);
  }

  void* praw = mmap(NULL, sizeof(PgpDaq::PgpCam), (PROT_READ|PROT_WRITE), (MAP_SHARED|MAP_LOCKED), fd, 0); 
  if (praw == MAP_FAILED) {
    perror("Map failed");
    return -1;
  }

  PgpDaq::PgpCam* p = (PgpDaq::PgpCam*)praw;
  printf("[%p]\n",p);
  printf("-- Core Axi Version --\n");
  printf("firmwareVersion    : %x\n", p->version);
  printf("scratch            : %x\n", p->scratch);
  printf("upTimeCnt          : %x\n", p->upTimeCnt);
  printf("dna                : %08x.%08x.%08x.%08x\n", 
         p->dna[3], p->dna[2], p->dna[1], p->dna[0] );

  uint32_t buildStr[64];
  for(int i=0; i<64; i++) {
    buildStr[i] = p->buildStr[i];
    printf("%x\r", buildStr[i]);
  }
  printf("buildString        : %s\n", reinterpret_cast<char*>(buildStr));

#define PRINTFIELD(name, addr, offset, mask) {                          \
    uint32_t reg;                                                       \
    printf("%20.20s :", #name);                                         \
    for(unsigned i=0; i<8; i++) {                                       \
      reg = q[(0x10000*i+addr)>>2];                                     \
      printf(" %8x", (reg>>offset)&mask);                               \
    }                                                                   \
    printf("\n"); }

    const uint32_t* q = reinterpret_cast<const uint32_t*>(&p->pgpLane[0]);
    printf("-- PgpMisc Registers --\n");
    PRINTFIELD(vcBlowoff, 0x0,  0, 0x1);
    PRINTFIELD(loopback , 0x0, 16, 0x7);
    PRINTFIELD(rxReset  , 0x0, 31, 0x1);
    PRINTFIELD(dropCount , 0x8, 0, 0xffffffff);
    PRINTFIELD(truncCount, 0xc, 0, 0xffffffff);

#undef PRINTFIELD
  
#define PRINTFIELD(name, addr, offset, mask) {                          \
    uint32_t reg;                                                       \
    printf("%20.20s :", #name);                                         \
    for(unsigned i=0; i<8; i++) {                                       \
      reg = q[(0x10000*i+0x8000+addr)>>2];                              \
      printf(" %8x", (reg>>offset)&mask);                               \
    }                                                                   \
    printf("\n"); }
#define PRINTBIT(name, addr, bit)  PRINTFIELD(name, addr, bit, 1)
#define PRINTREG(name, addr)       PRINTFIELD(name, addr,   0, 0xffffffff)
#define PRINTERR(name, addr)       PRINTFIELD(name, addr,   0, 0xf)
#define PRINTFRQ(name, addr) {                                          \
    uint32_t reg;                                                       \
    printf("%20.20s :", #name);                                         \
    for(unsigned i=0; i<8; i++) {                                       \
      reg = q[(0x10000*i+0x8000+addr)>>2];                              \
      printf(" %8.3f", float(reg)*1.e-6);                               \
    }                                                                   \
    printf("\n"); }

    q = reinterpret_cast<const uint32_t*>(&p->pgpLane[0]);
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

#undef PRINTFIELD
#define PRINTFIELD(name, addr, offset, mask) {                          \
    uint32_t reg;                                                       \
    printf("%20.20s :", #name);                                         \
    reg = q[(addr)>>2];                                                 \
    printf(" %8x\n", (reg>>offset)&mask);                               \
    }

    q = reinterpret_cast<const uint32_t*>(&p->sim);
    printf("sim @ %p\n", q);
    printf("-- PgpTxSim Registers --\n");
    PRINTFIELD(txEnable , 0x100,  0, 0x1);
    PRINTFIELD(txFixed  , 0x100,  8, 0x1);
    PRINTFIELD(txIntBase, 0x100,  9, 0xf);
    PRINTFIELD(txIntExp , 0x100, 13, 0x7);
    PRINTFIELD(txClear  , 0x100, 16, 0x1);
    PRINTFIELD(txReqDly , 0x100, 24, 0xf);
    PRINTFIELD(txReqMax , 0x100, 28, 0xf);
    PRINTFIELD(txLength , 0x104,  0, 0xffffffff);
    PRINTFIELD(overflow , 0,  0, 0xffffffff);

  close(fd);
}
