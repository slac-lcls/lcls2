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

using Pds::Cphw::Reg;
using Pds::Cphw::Reg64;
using Pds::Cphw::AmcTiming;
using Pds::Cphw::XBar;

extern int optind;

bool      _keepRunning = false;   // keep running on exit
unsigned  _partn = 0;             // link partition 

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -a <IP addr (dotted notation)> : Use network <IP>\n");
  printf("         -t <partition>                 : Link partition (default=0)\n");
  printf("         -k                             : Keep running on exit\n");
  printf("         -e                             : Enable DTI link\n");
  printf("         -m <message>                   : Insert 47-bit message (ULL)\n");
}

class Xpm {
public:
  AmcTiming _timing;
  uint32_t  _reserved[(0x80000000-sizeof(AmcTiming))>>2];
  //  uint32_t _reserved[0x80000000>>2];
  Reg      _paddr;
  Reg      _index; // [3:0]=partn, [9:4]=link, [15:10]=linkDbg, [19:16]=amc

  class LinkConfig {
  private:
    Reg    _control;
  public:
    void setup(bool lEnable) {
      // txDelay=0, partn=0, src=0, lb=F, txReset=F, rxReset=F, enable=F
      //      _control = 0;
      // txDelay=0, partn=0, src=0, lb=F, txReset=F, rxReset=F, enable=F
      _control = lEnable ? (1<<31) : 0;
    }
  } _linkConfig;
  
  class LinkStatus {
  private:
    Reg    _status;
  public:
    void status(bool& rxReady,
                bool& txReady,
                bool& isXpm,
                unsigned& rxErrs) {
      unsigned v = _status;
      rxErrs  = v&0xffff;
      txReady = v&(1<<17);
      rxReady = v&(1<<19);
      isXpm   = v&(1<<20);
    }
  } _linkStatus;

  Reg _pllCSR;
  
  class PartitionL0Config {
  private:
    Reg _control1;
    Reg _control2;
  public:
    void start(unsigned rate) {
      _control1 = 1;
      _control2 = (rate&0xffff) | 0x80000000;
      _control1 = 0x80010000;
    }
    void stop() {
      if (!_keepRunning) {
        _control1 = 0;
      }
    }
    void lockStats(bool v) {
      unsigned u = _control1;
      if (v) u &= ~(1<<31);
      else   u |=  (1<<31);
      _control1 = u;
    }
  } _partitionL0Config;
  
  class PartitionStatus {
  public:
    Reg64 _enabled;
    Reg64 _inhibited;
    Reg64 _ninput;
    Reg64 _ninhibited;
    Reg64 _naccepted;
    Reg64 _nacceptedl1;
  } _partitionStatus;

  Reg _reserved1[(104-76)>>2];

  Reg _pipelineDepth;
  Reg _msgHeader;
  Reg _msgPayload;

  Reg _reserved1b[(144-116)>>2];

  Reg _partitionSrcInhibits[32];

public:
  Xpm() {
    _index = (_partn & 0xf);
    _pipelineDepth = 90;
    usleep(100);
    message(1,90);
    usleep(10);
    message(0,0);
  }
  void message(unsigned header, unsigned payload) {
    _msgPayload = payload;
    _msgHeader  = (header&0x7fff) | (1<<15); // must be last
  }
  void start(unsigned rate, unsigned dtiMask) {
    //  Drive backplane
    _timing.xbar.setOut( XBar::BP, XBar::FPGA );

    // Configure dslink
    _index = (5<<4) | (_partn & 0xf);  // FP to RTM
    _linkConfig.setup(false);

    _index = (16<<4) | (_partn & 0xf);  // BP broadcast channel
    _linkConfig.setup(true);

    for(unsigned i=0; i<7; i++) {
      _index = ((17+i)<<4) | (_partn & 0xf);  // BP channel i
      _linkConfig.setup( dtiMask & (1<<i) );
    }

    // Setup partition
    _partitionL0Config.start(rate);
  }
  void stop() {
    _partitionL0Config.stop();
  }
  void stats(uint64_t* v,
             uint64_t* dv) {
#define FILL(s) {     \
      uint64_t q = _partitionStatus.s; \
      *dv++ = q - *v; \
      *v++  = q; }
    //    _partitionL0Config.lockStats(true);
    FILL(_enabled);
    FILL(_inhibited);
    FILL(_ninput);
    FILL(_ninhibited);
    FILL(_naccepted);
    //    _partitionL0Config.lockStats(false);
#undef FILL
  }
  void inhibits(uint32_t* v, 
                uint32_t* dv) {
    
#define FILL(s) {     \
      uint32_t q = _partitionSrcInhibits[s]; \
      *dv++ = q - *v; \
      *v++  = q; }
    for(unsigned i=0; i<32; i++) 
      FILL(i);
#undef FILL
  }
  void dsRecvs(uint16_t* v,
               uint16_t* dv,
               uint32_t& rxUp) {
    for(unsigned i=0; i<14; i++) {
      _index = (0+i)<<4;
      unsigned q; bool b0,b1,b2;
      _linkStatus.status(b0,b1,b2,q);
      dv[i] = q - v[i];
      v [i] = q;
      if (b0)
        rxUp |= (1<<i);
      else
        rxUp &= ~(1<<i);
    }
  }
  void bpRecvs(uint16_t* v,
               uint16_t* dv,
               uint32_t& rxUp) {
    for(unsigned i=0; i<16; i++) {
      _index = ((16+i)<<4) | (_partn & 0xf);
      unsigned q; bool b0,b1,b2;
      _linkStatus.status(b0,b1,b2,q);
      dv[i] = q - v[i];
      v [i] = q;
      if (b0)
        rxUp |= (1<<(16+i));
      else
        rxUp &= ~(1<<(16+i));
    }
  }
};

static void process(Xpm& xpm)
{
  static uint64_t stats[5], dstats[5];
  static const char* title[] = { "enabled   ",
                                 "inhibited ",
                                 "ninput    ",
                                 "ninhibited",
                                 "naccepted " };
  static uint32_t inh[32], dinh[32];
  static uint16_t dsrcv[16], dsdrcv[16];
  static uint16_t bprcv[16], bpdrcv[16];

  printf("link partition: %u\n", _partn);

  xpm.stats(stats,dstats);
  for(unsigned i=0; i<5; i++)
    printf("%s: %016llu [%010llu]\n", 
           title[i], (unsigned long long)stats[i], (unsigned long long)dstats[i]);

  xpm.inhibits(inh,dinh);
  for(unsigned i=0; i<32; i++)
    printf("%09u [%09u]%s", inh[i], dinh[i], (i&3)==3 ? "\n":"  ");

  uint32_t rxUp=0;
  xpm.dsRecvs(dsrcv,dsdrcv,rxUp);
  for(unsigned i=0; i<7; i++) {
    printf("  DS%i: %06u [%06u] (%04x)\n", i, dsrcv[i], dsdrcv[i], dsrcv[i+7]);
  }

  xpm.bpRecvs(bprcv,bpdrcv,rxUp);
  for(unsigned i=0; i<7; i++) {
    printf("  BP%i: %06u [%06u] (%04x)\n", i, bprcv[i], bpdrcv[i], bprcv[i+8]);
  }
  
  printf("rxUp %08x  paddr %08x\n", rxUp, unsigned(xpm._paddr));
  printf("===\n");
}

void sigHandler( int signal ) {
  Xpm* xpm = new (0)Xpm;
  xpm->stop();
  printf("\n====\n");
  process(*xpm);
  ::exit(signal);
}


int main(int argc, char** argv) {

  extern char* optarg;

  int c;

  const char* ip  = "10.0.1.102";
  unsigned fixed_rate = 6;
  unsigned dtiMask = 0;
  uint64_t msg=0;

  while ( (c=getopt( argc, argv, "a:d:em:r:t:kh")) != EOF ) {
    switch(c) {
    case 'a': ip = optarg; break;
    case 'd': dtiMask = strtoul(optarg,NULL,0); break;
    case 'e': dtiMask = 1; break;
    case 'm': msg = strtoull(optarg,NULL,0); msg |= 1ULL<<48; break;
    case 'r': fixed_rate = strtoul(optarg,NULL,0); break;
    case 't': _partn = strtoul(optarg, NULL, 0); break;
    case 'k': _keepRunning = true; break;
    case 'h': default:  usage(argv[0]); return 0;
    }
  }

  ::signal( SIGINT, sigHandler );

  //  Setup XPM
  Pds::Cphw::Reg::set(ip, 8192, 0);
  Xpm* xpm = new (0)Xpm;

  xpm->_timing.ring0.clear();
  xpm->_timing.ring0.enable(true);
  usleep(10);
  xpm->_timing.ring0.enable(false);
  xpm->_timing.ring0.dump();

  if (msg) xpm->message(msg>>32, msg&0xffffffff);

  xpm->start(fixed_rate, dtiMask);

  while(1) {
    sleep(1);
    process(*xpm);
  }

  return 0;
}
