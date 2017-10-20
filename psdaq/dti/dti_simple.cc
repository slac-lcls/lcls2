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

//#define NO_DSPGP

using Pds::Cphw::Reg;
using Pds::Cphw::Reg64;
using Pds::Cphw::AmcTiming;
using Pds::Cphw::RingBuffer;
using Pds::Cphw::XBar;

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
}

class Dti {
private:  // only what's necessary here
  AmcTiming _timing;
  uint32_t  _reserved[(0x80000000-sizeof(AmcTiming))>>2];
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
  } _usLink[7];
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

  uint32_t reserved2[(0x10000000-180)>>2];

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
  RingBuffer _ringb;

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
    _pgp[0]._countReset = 1;
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

    _pgp[0]._countReset = 0;
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
  void stats(uint32_t* v,
             uint32_t* dv,
             unsigned  us,
             unsigned  ds) {
    _linkIdx = (1<<31) + (us<<0) + (ds<<16);
#define UFILL(s) {     \
      uint32_t q = _usLinkStatus.s; \
      *dv++ = q - *v; \
      *v++  = q; }
    UFILL(_rxErrs);
    UFILL(_rxFull);
    { uint32_t q = _usLinkStatus._rxInh;
      *dv++ = (q - *v)&0xffffff;
      *v++  = q&0xffffff;
      *dv++ = ((q>>24)-*v)&0xf;
      *v++  = (q>>24)&0xf;
      *dv++ = ((q>>28)-*v)&0xf;
      *v++  = (q>>28)&0xf; }
    //    UFILL(_rxInh&0xffffff);
    //    UFILL(_rxInh&0xf000000>>24);
    //    UFILL(_rxInh&0xf0000000>>28);
    UFILL(_ibEvt);
#define DFILL(s) {     \
      uint32_t q = _dsLinkStatus.s; \
      *dv++ = q - *v; \
      *v++  = q; }
    DFILL(_rxErrs);
    DFILL(_rxFull);
    DFILL(_obSent);
    if (us) return;
#ifndef NO_DSPGP
#define PFILL(s) {          \
      uint32_t q = _pgp[0].s; \
      *dv++ = q - *v;         \
      *v++  = q;              \
      q = _pgp[1].s;          \
      *dv++ = q - *v;         \
      *v++  = q; }
#else
#define PFILL(s) {          \
      uint32_t q = _pgp[0].s; \
      *dv++ = q - *v;         \
      *v++  = q;              \
      *dv++ = 0; v++; }
#endif
    PFILL(_rxFrames);
    PFILL(_rxFrameErrs);
    PFILL(_txFrames);
    PFILL(_txFrameErrs);
    PFILL(_rxOpcodes);
    PFILL(_txOpcodes);
#define FILL(s) {               \
      uint32_t q = s;           \
      *dv++ = q - *v;           \
      *v++  = q; }
    FILL(_bpLinkSent);

    printf("link partition: %u\n", _partn);
    printf("linkUp   : %08x\n",unsigned(_linkUp));
  }
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
  char* endptr;
  while ( (c=getopt( argc, argv, "a:t:ru:f:kHh")) != EOF ) {
    switch(c) {
    case 'a': ip = optarg; break;
    case 't': _partn = strtoul(optarg, NULL, 0); break;
    case 'r': lRTM = true; break;
    case 'k': _keepRunning = true; break;
    case 'H': lHdrOnly = true; break;
    case 'u': upstream = strtoul(optarg, NULL, 0); break;
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

  ::signal( SIGINT, sigHandler );

  //  Setup DTI
  Pds::Cphw::Reg::set(ip, 8192, 0);
  Dti* dti = new (0)Dti(lRTM);
  dti->dumpb();
  dti->start(upstream, fwdmask,lHdrOnly);

  uint32_t stats[7][22], dstats[7][22];
  memset(stats, 0, sizeof(stats));
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

  memset(dstats,0,sizeof(dstats));

  unsigned fwdlink[7];
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
  

  while(1) {
    sleep(1);
    for(unsigned i=0; i<7; i++) {
      if (upstream & (1<<i)) {
        dti->stats(stats[i],dstats[i],i,fwdlink[i]);
      }
    }
    for(unsigned j=0; j<20; j++) {
      printf("%s: ", title[j]);
      for(unsigned i=0; i<7; i++) {
        if (upstream & (1<<i))
          printf(" %010u", stats[i][j]);
      }
      for(unsigned i=0; i<7; i++) {
        if (upstream & (1<<i))
          printf(" [%010u]", dstats[i][j]);
      }
      printf("\n");
    }
    printf("----\n");
  }

  return 0;
}
