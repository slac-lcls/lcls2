
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
#include <netdb.h>
#include <arpa/inet.h>

#include "DataDriver.h"
#include "Si570.hh"

using namespace std;

#define READREG(name,addr)                    \
  if (dmaReadRegister(ifd, addr, &reg)<0) {   \
      perror(#name);                          \
      return -1;                             \
    }
#define WRITEREG(name,addr)                                     \
  if (dmaWriteRegister(ifd, addr, reg)) {                       \
    perror(#name);                                              \
    return -1;                                                  \
  }
#define SETFIELD(name,addr,value,offset,mask) {                 \
    uint32_t reg;                                               \
    READREG(name,addr);                                         \
    reg &= ~(mask<<offset);                                     \
    reg |= (value&mask)<<offset;                                \
    WRITEREG(name,addr);                                        \
  }

int          fd  [2];

static int print_mig_lane(const char* name, int addr, int offset, int mask)
{
    const unsigned MIG_LANES = 0x00800080;
    printf("%20.20s", name);
    for(int i=0; i<2; i++) {
      int ifd = fd[i];
      if (ifd >= 0) {
        uint32_t reg; READREG( MIG_REG, MIG_LANES + addr);
        printf(" %8x", (reg >> offset) & mask);
      }
    }
    printf("\n");
    return 0;
}

static bool lInit = false;
static bool lReset = false;

static void check_program_clock(int ifd, const AxiVersion& vsn) 
{
  if (lInit || lReset) {
    if (vsn.userValues[2] == 0) {
      //  Set the I2C Mux
      dmaWriteRegister(ifd, 0x00e00000, (1<<2));
      //  Configure the Si570
      Kcu::Si570 s(ifd, 0x00e00800);
      if (lReset)
        s.reset();
      else
        s.program();
    }
    //  Reset the QPLL
    //uint32_t reg;
    dmaWriteRegister(ifd, 0x00a40024, 1);
    usleep(10);
    dmaWriteRegister(ifd, 0x00a40024, 0);
    usleep(10);
    //  Reset the Tx and Rx
    dmaWriteRegister(ifd, 0x00a40024, 6);
    usleep(10);
    dmaWriteRegister(ifd, 0x00a40024, 0);
  }
}

static void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -l <mask> -I\n");
  printf("Options: -l <mask>\n");
  printf("         -I (initialize 186MHz clock)\n");
  printf("         -R (reset to 156.25MHz clock)\n");
  printf("         -C (reset counters)\n");
  printf("         -Q (enable interrupts)\n");
}

int main (int argc, char **argv) {

  unsigned     base;
  int          lbmask = -1;
  bool         lCounterReset = false;
  bool         lUpdateId = false;
  bool         lIntEnable = false;
  int          c;

  while((c=getopt(argc,argv,"l:CIRQ"))!=-1) {
    switch(c) {
    case 'l': lbmask = strtoul(optarg,NULL,0); break;
    case 'C': lCounterReset = true; lUpdateId = true; break;
    case 'I': lInit = true; lUpdateId = true; break;
    case 'R': lReset = true; break;
    case 'Q': lIntEnable = true; break;
    default:  usage(argv[0]); return 1;
    }
  }

  if ( (fd[0] = open("/dev/datadev_1", O_RDWR)) <= 0 ) {
    cout << "Error opening /dev/datadev_1" << endl;
    //    return(1);
  }

  if ( (fd[1] = open("/dev/datadev_0", O_RDWR)) <= 0 ) {
    cout << "Error opening /dev/datadev_0" << endl;
    //    return(1);
  }

  { AxiVersion vsn;
    for(unsigned i=0; i<2; i++) {
      int ifd = fd[i];
      if (ifd >= 0 && axiVersionGet(ifd, &vsn)>=0) {
        printf("-- Core Axi Version --\n");
        printf("firmwareVersion : %x\n", vsn.firmwareVersion);
        printf("upTimeCount     : %u\n", vsn.upTimeCount);
        printf("deviceId        : %x\n", vsn.deviceId);
        printf("buildString     : %s\n", vsn.buildString); 
        printf("corePcie        : %c\n", (vsn.userValues[2] == 0) ? 'T':'F');
        printf("dmaSize         : %u\n", vsn.userValues[0]);
        printf("dmaClkFreq      : %u\n", vsn.userValues[4]);
        printf("axiAddrWidth    : %u\n", (vsn.userValues[7]>>24)&0xff);
        printf("axiDataWidth    : %u\n", (vsn.userValues[7]>>16)&0xff);
        printf("axiIdBits       : %u\n", (vsn.userValues[7]>> 8)&0xff);
        printf("axiLenBits      : %u\n", (vsn.userValues[7]>> 0)&0xff);
        check_program_clock(ifd, vsn);
      }
    }   
  }
  
#define PRINTFIELD(name, addr, offset, mask) {                  \
    uint32_t reg;                                               \
    printf("%20.20s :", #name);                                 \
    for(unsigned i=0; i<8; i++) {                               \
      int ifd = fd[i>>2];                                       \
      if (ifd>=0) {                                             \
        READREG(name,addr+base+(i&3)*0x10000);                  \
        printf(" %8x", (reg>>offset)&mask);                     \
      }                                                         \
    }                                                           \
    printf("\n"); }
#define PRINTBIT(name, addr, bit)  PRINTFIELD(name, addr, bit, 1)
#define PRINTREG(name, addr)       PRINTFIELD(name, addr,   0, 0xffffffff)
#define PRINTERR(name, addr)       PRINTFIELD(name, addr,   0, 0xf)
#define PRINTFRQ(name, addr) {                                          \
    uint32_t reg;                                                       \
    printf("%20.20s :", #name);                                         \
    for(unsigned i=0; i<8; i++) {                                       \
      if (fd[i>>2]>=0) {                                                \
        dmaReadRegister(fd[i>>2], addr+base+(i&3)*0x10000, &reg);       \
        printf(" %8.3f", float(reg)*1.e-6);                             \
      }                                                                 \
    }                                                                   \
    printf("\n"); }

  base = 0x00a08000;

  if (lbmask != -1) {
    unsigned reg;
    for(unsigned i=0; i<8; i++) {
      int ifd = fd[i>>2];
      if (ifd >= 0) {
        unsigned a = 0x08+base+(i&3)*0x10000;
        READREG(loopback, a);
        reg &= ~0x7;
        if (lbmask & (1<<i))
          reg |= 0x2;
        dmaWriteRegister(ifd,a,reg);
      }
    }
  }

  if (lCounterReset) {
    for(unsigned i=0; i<8; i++) {
      int ifd = fd[i>>2];
      if (ifd >= 0) {
        dmaWriteRegister(ifd, base, 1);
        usleep(10);
        dmaWriteRegister(ifd, base, 0);
      }
    }
    usleep(10000);
  }

  //
  //  Update ID advertised on timing link
  //
  if (lUpdateId) {
    struct addrinfo hints;
    struct addrinfo* result;

    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_INET;       /* Allow IPv4 or IPv6 */
    hints.ai_socktype = SOCK_DGRAM; /* Datagram socket */
    hints.ai_flags = AI_PASSIVE;    /* For wildcard IP address */

    char hname[64];
    gethostname(hname,64);
    int s = getaddrinfo(hname, NULL, &hints, &result);
    if (s != 0) {
      fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(s));
      exit(EXIT_FAILURE);
    }

    sockaddr_in* saddr = (sockaddr_in*)result->ai_addr;

    unsigned id = 0xfb000000 | 
      (ntohl(saddr->sin_addr.s_addr)&0xffff);

    for(unsigned i=0; i<8; i++) {
      int ifd = fd[i>>2];
      if (ifd >= 0)
        dmaWriteRegister(ifd, 0x00a40010+4*(i&3), id | (i<<16));
    }
  }

  {
    printf("\n-- migLane Registers --\n");
    print_mig_lane("blockSize  ", 0, 0, 0x1f);
    print_mig_lane("blocksPause", 4, 8, 0x3ff);
    print_mig_lane("blocksFree ", 8, 0, 0x1ff);
    print_mig_lane("blocksQued ", 8,12, 0x1ff);
    print_mig_lane("writeQueCnt",12, 0, 0xff);
    print_mig_lane("wrIndex    ",16, 0, 0x1ff);
    print_mig_lane("wcIndex    ",20, 0, 0x1ff);
    print_mig_lane("rdIndex    ",24, 0, 0x1ff);
    print_mig_lane("ilvStatus  ",288, 0, 0xffffffff);
  }

#define PRINTCLK(name, addr) {                                          \
    uint32_t reg;                                                       \
    printf("%20.20s :", #name);                                         \
    for(unsigned i=0; i<2; i++) {                                       \
      int ifd = fd[i];                                                  \
      if (ifd >= 0) {                                                   \
        dmaReadRegister(ifd, addr, &reg);                               \
        printf(" %8.3f MHz", float(reg&0x1fffffff)*1.e-6);              \
        printf(" (%s)", (reg&(1<<31)) ? "Locked":"Not Locked");         \
      }                                                                 \
    }                                                                   \
    printf("\n");                                                       \
  }                                                                     \

  PRINTCLK(axilClk, 0x800100);
  PRINTCLK(sysClk , 0x800104);
  PRINTCLK(clk200 , 0x800108);
  PRINTCLK(pgpClk , 0x80010c);
  { uint32_t v;
    printf("qPllLock:");
    for(unsigned i=0; i<2; i++) {
      int ifd = fd[i];
      if (ifd >= 0) {
        dmaReadRegister(ifd, 0x00a40020, &v);
        printf(" %s", (v&1) ? "Locked" : "Not Locked");
      }
    }
    printf("\n");
  }
  { uint32_t v;
    printf("phyReset:");
    for(unsigned i=0; i<2; i++) {
      int ifd = fd[i];
      if (ifd >= 0) {
        dmaReadRegister(ifd, 0x00a40024, &v);
        printf(" %x", v&7);
      }
    }
    printf("\n");
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
  PRINTREG(locOflowCnt, 0xB0);
  PRINTFIELD(locPause   , 0x8c, 16, 0xffff);
  PRINTREG(txFrameCnt , 0x90);
  PRINTERR(txFrameErrCnt, 0x94);
  PRINTFRQ(txClkFreq  , 0x9c);
  PRINTERR(txOpCodeCnt, 0xa0);
  PRINTREG(txOpCodeLst, 0xa4);

#define PRINTID(name) {                         \
    uint64_t v;                                 \
    uint32_t reg;                               \
    printf("%20.20s :", #name);                 \
    for(unsigned i=0; i<8; i++) {               \
      int ifd = fd[i>>2];                       \
      if (ifd>=0) {                                             \
        READREG(name,0x38+base+(i&3)*0x10000);                  \
        v = reg;                                                \
        v <<= 32;                                               \
        READREG(name,0x34+base+(i&3)*0x10000);                  \
        v |= reg;                                               \
        unsigned id = (v>>16)&0xffffffff;                       \
        printf(" %8x", id);                                     \
      }                                                         \
    }                                                           \
    printf("\n"); }

  PRINTID(rxLinkId);

  base = 0x00a40000;
#undef PRINTID
#define PRINTID(name, addr) {                                   \
    uint32_t reg;                                               \
    printf("%20.20s :", #name);                                 \
    for(unsigned i=0; i<8; i++) {                               \
      int ifd = fd[i>>2];                                       \
      if (ifd>=0) {                                             \
        READREG(name,addr+base+(i&3)*4);                        \
        printf(" %8x", reg);                                    \
      }                                                         \
    }                                                           \
    printf("\n"); }
  PRINTID(txLinkId, 0x10);

  if (lIntEnable) {
    if (fd[0]>=0)
      ioctl(fd[0], 0x2002, 0);
    if (fd[1]>=0)
      ioctl(fd[1], 0x2002, 0);
  }

  close(fd[0]);
  close(fd[1]);
}
