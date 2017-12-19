/**
 ** pgpdaq
 **
 **   Manage XPM and DTI to trigger and readout pgpcard (dev03)
 **
 **/

#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>
#include <arpa/inet.h>
#include <signal.h>
#include <new>

#include "psdaq/cphw/Reg.hh"
#include "psdaq/cphw/Reg64.hh"
#include "psdaq/cphw/AmcTiming.hh"
#include "psdaq/cphw/RingBuffer.hh"
#include "psdaq/cphw/XBar.hh"
#include "psdaq/cphw/HsRepeater.hh"

#define NO_USPGP
#define NO_DSPGP

using Pds::Cphw::Reg;
using Pds::Cphw::Reg64;
using Pds::Cphw::AmcTiming;
using Pds::Cphw::RingBuffer;
using Pds::Cphw::XBar;
using Pds::Cphw::HsRepeater;

extern int optind;
bool _keepRunning = false;    // keep running on exit
unsigned _partn = 0;

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -a <IP addr (dotted notation)> : Use network <IP>\n");
  printf("         -t <partition>                 : Upstream link partition (default=0)\n");
  printf("         -u <upstream mask>             : Bit mask of upstream links (default=1)\n");
  printf("         -f <forward mask>              : Comma separated list (no spaces) of bit masks of downstream links, one entry per enabled upstream link (default=1)\n");
  printf("         -k                             : Keep running on exit\n");
  printf("         -L <fname>                     : Load configuration from file\n");
}

class LinkStats {
public:
  LinkStats() :
 usRxErrs(0),
 usRxFull(0),
 usRxInh(0),
 usWrFifoD(0),
 usRdFifoD(0),
 usIbEvt(0),
 dsRxErrs(0),
 dsRxFull(0),
 dsObSent(0),
 usRxFrames(0),
 dsRxFrames(0),
 usRxFrErrs(0),
 dsRxFrErrs(0),
 usTxFrames(0),
 dsTxFrames(0),
 usTxFrErrs(0),
 dsTxFrErrs(0),
 usRxOpCodes(0),
 dsRxOpCodes(0),
 usTxOpCodes(0),
 dsTxOpCodes(0),
 bpLinkSent(0) {}
public:
  unsigned usRxErrs;
  unsigned usRxFull;
  unsigned usRxInh;
  unsigned usWrFifoD;
  unsigned usRdFifoD;
  unsigned usIbEvt;
  unsigned dsRxErrs;
  unsigned dsRxFull;
  unsigned dsObSent;
  unsigned usRxFrames;
  unsigned dsRxFrames;
  unsigned usRxFrErrs;
  unsigned dsRxFrErrs;
  unsigned usTxFrames;
  unsigned dsTxFrames;
  unsigned usTxFrErrs;
  unsigned dsTxFrErrs;
  unsigned usRxOpCodes;
  unsigned dsRxOpCodes;
  unsigned usTxOpCodes;
  unsigned dsTxOpCodes;
  unsigned bpLinkSent;
public:
  unsigned stat(unsigned j) const { return (&usRxErrs)[j]; }
};

class Dti {
private:  // only what's necessary here
  AmcTiming _timing;
  uint32_t _reserved_AT[(0x09000000-sizeof(AmcTiming))>>2];
public:
  HsRepeater hsRepeater[6];
private:
  uint32_t _reserved_HR[(0x77000000-sizeof(hsRepeater))>>2];

  class UsLinkControl {
  private:
    Reg _control;
    Reg _dataSrc;
    Reg _dataType;
    uint32_t _reserved;
  public:
    void enable(unsigned fwdmask, bool hdrOnly=false) {
      unsigned control;
      if (fwdmask) {
        // enable=T, tagEnable=F, L1Enable=F, fwdMode=RR
        control = 1 | ((_partn&0xf)<<4) | ((fwdmask&0x1fff)<<16);
        if (hdrOnly) 
          control |= (1<<3);
      }
      else {
        control = 0;
      }
      _control = control;
    }
  } _usLink[7];  // 0x80000000
  Reg _linkUp; // [6:0]=us, [15]=bp, [28:16]=ds

  Reg _linkIdx; // [3:0]=us, [19:16]=ds, [30]=countReset, [31=countUpdate

  Reg _bpLinkSent;

  uint32_t _reserved1;

  class UsLinkStatus {
  public:
    Reg _rxErrs;
    Reg _rxFull;
    Reg _rxInh;
    Reg _ibEvt;
  } _usLinkStatus;
  class DsLinkStatus {
  public:
    Reg _rxErrs;
    Reg _rxFull;
    Reg _obSent;
    uint32_t _reserved;
  } _dsLinkStatus;

  uint32_t reserved2a[4];

  Reg _qpll_bpUpdate;
  Reg _monClk[4];
public:
  unsigned qpllLock() const { return unsigned(_qpll_bpUpdate)&3; }
  unsigned monClk(unsigned i) const { return unsigned(_monClk[i]); }
private:
  uint32_t reserved2[(0x10000000-196)>>2];

#if 0
  class Pgp2bAxi {
  public:
    Reg      _countReset;
    uint32_t _reserved[16];
    Reg      _rxFrameErrs;
    Reg      _rxFrames;
    uint32_t _reserved2[4];
    Reg      _txFrameErrs;
    Reg      _txFrames;
    uint32_t _reserved3[5];
    Reg      _txOpcodes;
    Reg      _rxOpcodes;
    uint32_t _reserved4[0x80>>2];
  } _pgp[2];

  uint32_t reserved3[(0x10000000-0x200)>>2];
#endif
  uint32_t reserved3[0x10000000>>2];

  RingBuffer _ringb;  // 0xA0000000

  uint32_t reserved4[(0x10000000-sizeof(_ringb))>>2];

  class Pgp3Us {
  public:
    class Pgp3Axil {
    public:
      uint32_t countReset;
      uint32_t autoStatus;
      uint32_t loopback;
      uint32_t skpInterval;
      uint32_t rxStatus; // phyRxActive, locLinkReady, remLinkReady
      uint32_t cellErrCnt;
      uint32_t linkDownCnt;
      uint32_t linkErrCnt;
      uint32_t remRxOflow; // +pause
      uint32_t rxFrameCnt;
      uint32_t rxFrameErrCnt;
      uint32_t rxClkFreq;
      uint32_t rxOpCodeCnt;
      uint32_t rxOpCodeLast;
      uint32_t rxOpCodeNum;
      uint32_t rsvd_3C;
      uint32_t rsvd_40[0x10];
      // tx
      uint32_t cntrl; // flowCntDis, txDisable
      uint32_t txStatus; // phyTxActive, linkReady
      uint32_t rsvd_88;
      uint32_t locStatus; // locOflow, locPause
      uint32_t txFrameCnt;
      uint32_t txFrameErrCnt;
      uint32_t rsvd_98;
      uint32_t txClkFreq;
      uint32_t txOpCodeCnt;
      uint32_t txOpCodeLast;
      uint32_t txOpCodeNum;
      uint32_t rsvd_AC;
      uint32_t rsvd_B0[0x14];
      uint32_t reserved[0x700>>2];
    public:
    } _pgp;
    class Drp {
    public:
      uint32_t reserved[0x800>>2];
    };
  } _pgpUs[7];

public:
  void dumpPgp() const {
#define PRINTFIELD(name, addr, offset, mask) {                          \
      uint32_t reg;                                                     \
      printf("%20.20s :", #name);                                       \
      for(unsigned i=0; i<7; i++) {                                     \
        const Reg* r = reinterpret_cast<const Reg*>((char*)(&_pgpUs[i]._pgp)+addr); \
        reg = *r;                                                       \
        printf(" %8x [%p]", (reg>>offset)&mask,r);                       \
      }                                                                 \
      printf("\n"); }
#define PRINTBIT(name, addr, bit)  PRINTFIELD(name, addr, bit, 1)
#define PRINTREG(name, addr)       PRINTFIELD(name, addr,   0, 0xffffffff)
#define PRINTERR(name, addr)       PRINTFIELD(name, addr,   0, 0xf)
#define PRINTFRQ(name, addr) {                                  \
      uint32_t reg;                                             \
      printf("%20.20s :", #name);                               \
      for(unsigned i=0; i<7; i++) {                                     \
        reg = *reinterpret_cast<const Reg*>((char*)(&_pgpUs[i]._pgp)+addr); \
        printf(" %8.3f", float(reg)*1.e-6);                     \
      }                                                         \
      printf("\n"); }

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
  }

public:
  Dti(bool lRTM) {
    //  XPM connected through RTM
    _timing.xbar.setOut( XBar::RTM0, XBar::FPGA );
    _timing.xbar.setOut( XBar::FPGA, lRTM ? XBar::RTM0 : XBar::BP );

    _qpll_bpUpdate = 100<<16;
  }

  void dumpb()
  {
    RingBuffer* b = &_ringb;
    printf("RingB @ %p\n",b);
    b->enable(false);
    b->clear();
    b->enable(true);
    usleep(1000);
    b->enable(false);
    b->dump(18);
    printf("\n");
  }
  void start(unsigned umask, unsigned* fmask, bool hdrOnly=false)
  {
    _linkIdx = 1<<30;
#ifndef NO_USPGP
    _pgp[0]._countReset = 1;
#endif
#ifndef NO_DSPGP
    _pgp[1]._countReset = 1;
#endif

    for(unsigned i=0; umask; i++) {
      if (umask & (1<<i)) {
        printf("Enable usLink[%u] fwdmask[%x]\n", i,fmask[i]);
        _usLink[i].enable(fmask[i], hdrOnly);
        umask &= ~(1<<i);
      }
      else
        _usLink[i].enable(0);
    }

#ifndef NO_USPGP
    _pgp[0]._countReset = 0;
#endif
#ifndef NO_DSPGP
    _pgp[1]._countReset = 0;
#endif
    _linkIdx = 1<<31;
  }
  void stop()
  {
    if (!_keepRunning) {
      for(unsigned i=0; i<7; i++)
        _usLink[i].enable(0);
    }
  }
  void stats(LinkStats& v,
             LinkStats& dv,
             unsigned  us,
             unsigned  ds) {
    _linkIdx = (1<<31) + (us<<0) + (ds<<16);
#define UFILL(s,t,op) {                         \
      uint32_t q = _usLinkStatus.s op;          \
      dv.t = q - v.t;                           \
      v. t = q; }
    UFILL(_rxErrs, usRxErrs, &0xffff);
    UFILL(_rxFull, usRxFull, );
    { uint32_t q = _usLinkStatus._rxInh;
      dv.usRxInh = (q - v.usRxInh)&0xffffff;
      v .usRxInh =  q&0xffffff;
      dv.usWrFifoD = ((q>>24)-v.usWrFifoD)&0xf;
      v .usWrFifoD = (q>>24)&0xf;
      dv.usRdFifoD = ((q>>28)-v.usRdFifoD)&0xf;
      v .usRdFifoD = (q>>28)&0xf; }
    //    UFILL(_rxInh&0xffffff);
    //    UFILL(_rxInh&0xf000000>>24);
    //    UFILL(_rxInh&0xf0000000>>28);
    UFILL(_ibEvt, usIbEvt, );
#define DFILL(s, t, op) {                       \
      uint32_t q = _dsLinkStatus.s op;          \
      dv.t = q - v.t;                           \
      v .t = q; }
    DFILL(_rxErrs, dsRxErrs, &0xffff);
    DFILL(_rxFull, dsRxFull, );
    DFILL(_obSent, dsObSent, );
    if (us==0) {
#ifndef NO_DSPGP
#define PFILL(s, t, u) {      \
      uint32_t q = _pgp[0].s; \
      dv.t = q - v.t;         \
      v .t = q;              \
      q = _pgp[1].s;          \
      dv.u = q - v.u;         \
      v .u = q; }
#else
#ifndef NO_USPGP
#define PFILL(s, t, u) {        \
      uint32_t q = _pgp[0].s; \
      dv.t = q - v.t;         \
      v. t = q;              \
      dv.u = 0; }
#else
#define PFILL(s, t, u) {        \
      dv.t = 0; \
      dv.u = 0; }
#endif
#endif
      PFILL(_rxFrames   , usRxFrames, dsRxFrames);
      PFILL(_rxFrameErrs, usRxFrErrs, dsRxFrErrs);
      PFILL(_txFrames   , usTxFrames, dsTxFrames);
      PFILL(_txFrameErrs, usTxFrErrs, dsTxFrErrs);
      PFILL(_rxOpcodes  , usRxOpCodes, dsRxOpCodes);
      PFILL(_txOpcodes  , usTxOpCodes, dsTxOpCodes);
    }
#define FILL(s, t) {            \
      uint32_t q = s;           \
      dv.t = q - v.t;           \
      v .t = q; }
    FILL(_bpLinkSent, bpLinkSent);
  }

  unsigned linkUp() const { return unsigned(_linkUp); }
};

void sigHandler( int signal ) {
  Dti* m = (Dti*)0;
  m->stop();
  ::exit(signal);
}

int main(int argc, char** argv) {

  extern char* optarg;

  int c;

  const char* ip  = "10.0.1.103";
  bool lRTM = false;
  bool lHdrOnly = false;
  unsigned upstream = 1;
  unsigned fwdmask[8]={1,0,0,0,0,0,0,0};
  const char* ifile=0;
  char* endptr;
  while ( (c=getopt( argc, argv, "a:t:ru:f:L:kHh")) != EOF ) {
    switch(c) {
    case 'a': ip = optarg; break;
    case 't': _partn = strtoul(optarg, NULL, 0); break;
    case 'r': lRTM = true; break;
    case 'k': _keepRunning = true; break;
    case 'H': lHdrOnly = true; break;
    case 'u': upstream = strtoul(optarg, NULL, 0); break;
    case 'L': ifile = optarg; break;
    case 'f': 
      endptr = optarg;
      for(unsigned i=0; i<7; i++) {
        if (upstream & (1<<i)) {
          fwdmask[i] = strtoul(endptr, &endptr, 0);
          endptr++;
        }
        else
          fwdmask[i] = 0;
      } 
      break;
    case 'h': default:  usage(argv[0]); return 0;
    }
  }

  ::signal( SIGINT , sigHandler );
  ::signal( SIGABRT, sigHandler );
  ::signal( SIGKILL, sigHandler );

  //  Setup DTI
  Pds::Cphw::Reg::set(ip, 8192, 0);

  //  Program the crossbar to pull timing off the backplane
  Pds::Cphw::AmcTiming* tim = new (0)Pds::Cphw::AmcTiming;
  tim->xbar.setOut( Pds::Cphw::XBar::FPGA, Pds::Cphw::XBar::BP );

  Dti* dti = new (0)Dti(lRTM);

  if (ifile) {
    FILE* f = fopen(ifile,"r");
    for(unsigned i=0; i<6; i++)
      dti->hsRepeater[i].load(f);
    fclose(f);
  }

  dti->dumpb();
  dti->dumpPgp();
  dti->start(upstream, fwdmask,lHdrOnly);

  LinkStats stats[7], dstats[7];

  static const char* title[] = 
    { "usRxErrs   ",
      "usRxFull   ",
      "usRxInh    ",
      "usWrFifoD  ",
      "usRdFifoD  ",
      "usIbEvt    ",
      "dsRxErrs   ",
      "dsRxFull   ",
      "dsObSent   ",
      "usRxFrames ",
      "dsRxFrames ",
      "usRxFrErrs ",
      "dsRxFrErrs ",
      "usTxFrames ",
      "dsTxFrames ",
      "usTxFrErrs ",
      "dsTxFrErrs ",
      "usRxOpcodes",
      "dsRxOpcodes",
      "usTxOpcodes",
      "dsTxOpcodes",
      "bpLinkSent " };

  unsigned fwdlink[7];
  memset(fwdlink,0,sizeof(fwdlink));
  for(unsigned i=0; i<7; i++)
    if (upstream & (1<<i)) 
      for(unsigned j=0; j<7; j++)
        if (fwdmask[i]&(1<<j)) {
          fwdlink[i]=j;
          break;
        }

  printf("links: %x :", upstream);
  for(unsigned i=0; i<7; i++)
    printf(" %x",fwdmask[i]);
  printf(" : ");
  for(unsigned i=0; i<7; i++)
    printf(" %x",fwdlink[i]);
  printf("\n");
  
  printf("qpllLock : %x\n", dti->qpllLock());
  printf("monClk   : %f(%x) %f(%x) %f(%x) %f(%x)\n",
         float(dti->monClk(0)&0x1fffffff)*1.e-6, dti->monClk(0)>>29,
         float(dti->monClk(1)&0x1fffffff)*1.e-6, dti->monClk(1)>>29,
         float(dti->monClk(2)&0x1fffffff)*1.e-6, dti->monClk(2)>>29,
         float(dti->monClk(3)&0x1fffffff)*1.e-6, dti->monClk(3)>>29);

  while(1) {
    sleep(1);
    printf("link partition: %u\n", _partn);
    printf("linkUp   : %08x\n",dti->linkUp());
    for(unsigned i=0; i<7; i++) {
      if (upstream & (1<<i)) {
        dti->stats(stats[i],dstats[i],i,fwdlink[i]);
      }
    }
    for(unsigned j=0; j<20; j++) {
      printf("%s: ", title[j]);
      for(unsigned i=0; i<7; i++) {
        if (upstream & (1<<i))
          printf(" %010u", stats[i].stat(j));
      }
      for(unsigned i=0; i<7; i++) {
        if (upstream & (1<<i))
          printf(" [%010u]", dstats[i].stat(j));
      }
      printf("\n");
    }
    printf("----\n");
  }

  return 0;
}
