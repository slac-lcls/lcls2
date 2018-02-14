
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>
#include <arpa/inet.h>
#include <poll.h>
#include <signal.h>
#include <new>

#include "psdaq/hsd/Module.hh"
#include "psdaq/hsd/QABase.hh"
#include "psdaq/hsd/Globals.hh"
#include "psdaq/mmhw/AxiVersion.hh"
#include "psdaq/hsd/TprCore.hh"
#include "psdaq/hsd/FexCfg.hh"
#include "psdaq/hsd/Pgp2bAxi.hh"

#include <string>
#include <vector>

extern int optind;

using namespace Pds::HSD;

class DmaStats {
public:
  DmaStats() : _values(4) {
    for(unsigned i=0; i<_values.size(); i++)
      _values[i]=0;
  }
  DmaStats(const QABase& o) : _values(4) {
    frameCount   () = o.countEnable;
    pauseCount   () = o.countInhibit;
  }

public:
  static const char** names();
  std::vector<unsigned> values() const { return _values; }
public:
  unsigned& frameCount   () { return _values[0]; }
  unsigned& pauseCount   () { return _values[1]; }
  unsigned& overflowCount() { return _values[2]; }
  unsigned& idleCount    () { return _values[3]; }
private:
  std::vector<unsigned> _values;
};  

const char** DmaStats::names() {
  static const char* _names[] = {"frameCount",
                                 "pauseCount",
                                 "overflowCount",
                                 "idleCount" };
  return _names;
}

class PgpStats {
public:
  PgpStats(Module& m, unsigned chmask) : 
    _axi   (reinterpret_cast<Pgp2bAxi*>((char*)m.reg()+0x90000)),
    _chmask(chmask)
  {}
public:
  void dump() {
#define GET_STAT(s, delta) {                            \
      printf("%12.12s:", #s);                           \
      for(unsigned i=0; i<4; i++)                       \
        if (_chmask & (1<<i)) {                         \
          unsigned v = _axi[i].s;                       \
          _dstats[c] = (v-_stats[c]);                   \
          _stats [c] = v;                               \
          printf(" %08x", delta ? _dstats[c] : v);      \
          c++; }                                        \
      printf("\n"); }
    unsigned c=0;
    GET_STAT(_txFrames, true);
    GET_STAT(_txFrameErrs, true);
    GET_STAT(_rxOpcodes, true);
    GET_STAT(_lastRxOpcode, false);
  }
private:
  Pgp2bAxi* _axi;
  unsigned  _chmask;
  unsigned  _stats [16];
  unsigned  _dstats[16];
};

#define DUMP_STAT(stat,strm,op) {                                  \
    printf("%8.8s:s%u:", #stat, strm);                          \
    for(unsigned i=0; i<4; i++)                                 \
      if (_chmask & (1<<i))                                     \
        printf(" %04x", _fexcfg[i]._base[strm]._free op);       \
    printf("\n"); }

class FexStats {
public:
  FexStats(Module& m, unsigned chmask, unsigned smask) :
    _fexcfg(m.fex()),
    _chmask(chmask),
    _smask (smask) 
  {}
public:
  void dump() {
    for(unsigned j=0; j<4; j++)
      if (_smask & (1<<j)) {
        DUMP_STAT(free, j, &0xffff);
        DUMP_STAT(nfree, j, >>16&0x1f );
      }
  }
private:
  FexCfg* _fexcfg;
  unsigned _chmask;
  unsigned _smask;
};

class CacheDump {
public:
  CacheDump(Module& m) :
    _base  (*reinterpret_cast<QABase*>((char*)m.reg()+0x80000))
  {}
public:
  void dump() {
    static const char* stateName[] = {"EMPTY","OPEN","CLOSED","READING"};
    static const char* trigName [] = {"WAIT","ACCEPT","REJECT"};
    unsigned state[16];
    unsigned addr [16];
    for(unsigned i=0; i<16; i++) {
      _base.cacheSel = i;
      usleep(1);
      state[i] = _base.cacheState;
      addr [i] = _base.cacheAddr;
    }
#define PRINT_CACHE(title, fmt, arg) {                  \
      printf("%12.12s:", #title);                       \
      for(unsigned i=0; i<16; i++)                      \
        printf(fmt, arg);                               \
      printf("\n"); }
    PRINT_CACHE(cacheState, " %6.6s", stateName[state[i]&0xf]);    
    PRINT_CACHE(trigState , " %6.6s", trigName [state[i]>>4&0xf]);    
    PRINT_CACHE(begAddr   , "   %04x", addr[i]&0xffff);
    PRINT_CACHE(endAddr   , "   %04x", addr[i]>>16&0xffff);
  }
private:
  QABase& _base;
};

template <class T> class RateMonitor {
public:
  RateMonitor() {}
  RateMonitor(const T& o) {
    clock_gettime(CLOCK_REALTIME,&tv);
    _t = o;
  }
  RateMonitor<T>& operator=(const RateMonitor<T>& o) {
    tv = o.tv;
    _t = o._t;
    return *this;
  }
public:
  void dump(const RateMonitor<T>& o) {
    double dt = double(o.tv.tv_sec-tv.tv_sec)+1.e-9*(double(o.tv.tv_nsec)-double(tv.tv_nsec));
    for(unsigned i=0; i<_t.values().size(); i++)
      printf("%10u %15.15s [%10u] : %g\n",
             _t.values()[i],
             _t.names()[i],
             o._t.values()[i]-_t.values()[i],
             double(o._t.values()[i]-_t.values()[i])/dt);
  }
private:
  timespec tv;
  T _t;
};

static Module* reg=0;

void sigHandler( int signal ) {
  if (reg) {
    reg->stop();
    reinterpret_cast<QABase*>((char*)reg->reg()+0x80000)->dump();
  }

  ::exit(signal);
}

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -d <dev id>\n");
  printf("         -c <channelmask>\n");
  printf("         -s <streammask>\n");
  printf("         -F <min,max,pre,post> (FEX configuration parameters) [default 0x100,0x300,2,3]\n");
  printf("         -r <rate>\n");
  printf("         -l <length>\n");
  printf("         -p <partition>\n");
  printf("         -t <testpattern>\n");
  printf("         -R (reset)\n");
  printf("         -L (flip loopback settings)\n");
}

int main(int argc, char** argv) {

  extern char* optarg;
  char* endptr;

  char qadc='a';
  int c;
  bool lUsage = false;
  bool lReset = false;
  bool lLoopback = false;
  int  pattern = -1;
  int  rate    = -1;
  int  partition = -1;
  int  length  = 32;
  unsigned channelMask = 0xf;
  unsigned streamMask  = 0x3;
  unsigned fexMin  = 0x100;
  unsigned fexMax  = 0x300;
  unsigned fexPre  = 2;
  unsigned fexPost = 3;

  while ( (c=getopt( argc, argv, "c:s:d:r:l:p:t:F:LRh")) != EOF ) {
    switch(c) {
    case 'c':
      channelMask = strtoul(optarg,&endptr,0);
      break;
    case 's':
      streamMask = strtoul(optarg,&endptr,0);
      break;
    case 'd':
      qadc = optarg[0];
      break;
    case 'r':
      rate = atoi(optarg);
      break;
    case 'l':
      length = strtoul(optarg,&endptr,0);
      break;
    case 'p':
      partition = strtoul(optarg,&endptr,0);
      break;
    case 't':
      pattern = strtoul(optarg,&endptr,0);
      break;
    case 'F':
      fexMin = strtoul(optarg,&endptr,0);
      if (*endptr) {
        fexMax = strtoul(endptr+1,&endptr,0);
        if (*endptr) {
          fexPre = strtoul(endptr+1,&endptr,0);
          if (*endptr)
            fexPost = strtoul(endptr+1,&endptr,0);
        }
      }
      break;
    case 'L':
      lLoopback = true;
      break;
    case 'R':
      lReset = true;
      break;
    case '?':
    default:
      lUsage = true;
      break;
    }
  }

  if (lUsage) {
    usage(argv[0]);
    exit(1);
  }

  char devname[16];
  sprintf(devname,"/dev/qadc%c",qadc);
  int fd = open(devname, O_RDWR);
  if (fd<0) {
    perror("Open device failed");
    return -1;
  }

  Module* p = Module::create(fd);

  if (lReset) {
    reinterpret_cast<QABase*>((char*)p->reg()+0x80000)->resetFbPLL();
    usleep(1000000);
    reinterpret_cast<QABase*>((char*)p->reg()+0x80000)->resetFb ();
    reinterpret_cast<QABase*>((char*)p->reg()+0x80000)->resetDma();
    usleep(1000000);
  }

  if (lLoopback) {
    Pgp2bAxi* pgp = reinterpret_cast<Pgp2bAxi*>((char*)p->reg()+0x90000);
    for(unsigned i=0; i<4; i++)
      pgp[i]._loopback ^= 2;
    for(unsigned i=0; i<4; i++)
      pgp[i]._rxReset = 1;
    usleep(10);
    for(unsigned i=0; i<4; i++)
      pgp[i]._rxReset = 0;
    usleep(100);
  }

  p->dumpPgp();

  if (rate<0 && partition<0) return 0;

  p->disable_test_pattern();
  if (pattern >= 0)
    p->enable_test_pattern((Module::TestPattern)pattern);

  p->sample_init(32+48*length, 0, 0);
  p->setAdcMux( false, channelMask );

  if (partition >= 0)
    p->trig_daq(partition);
  else
    p->trig_lclsii( rate );

  //  Configure FEX
  FexCfg* fex = p->fex();
  for(unsigned i=0; i<4; i++) {
    if ((1<<i)&channelMask) {
      fex[i]._base[0].setGate(4,length);
      fex[i]._base[0].setFull(0xc00,4);
      fex[i]._base[0]._prescale=0;
      fex[i]._base[1].setGate(4,length);
      fex[i]._base[1].setFull(0xc00,4);
      fex[i]._base[1]._prescale=0;
      fex[i]._stream[1].parms[0].v = fexMin;
      fex[i]._stream[1].parms[1].v = fexMax;
      fex[i]._stream[1].parms[2].v = fexPre;
      fex[i]._stream[1].parms[3].v = fexPost;
      fex[i]._streams = streamMask;
    }
    else
      fex[i]._streams = 0;
  }
  
#define PRINT_FEX_FIELD(title,arg,op) {                       \
    printf("%12.12s:",title);                                 \
    for(unsigned i=0; i<4; i++) {                             \
      if (((1<<i)&channelMask)==0) continue;                  \
      printf(" %u/%u",                                        \
             fex[i]._base[0].arg op,                          \
             fex[i]._base[1].arg op);                         \
    }                                                         \
    printf("\n"); }                             
  
  PRINT_FEX_FIELD("GateBeg", _gate, &0x3fff);
  PRINT_FEX_FIELD("GateLen", _gate, >>16&0x3fff);
  PRINT_FEX_FIELD("FullRow", _full, &0xffff);
  PRINT_FEX_FIELD("FullEvt", _full, >>16&0x1f);
  PRINT_FEX_FIELD("Prescal", _prescale, &0x3ff);

  printf("streams:");
  for(unsigned i=0; i<4; i++) {
    if (((1<<i)&channelMask)==0) continue;
    printf(" %2u", fex[i]._streams &0xf);
  }
  printf("\n");

#undef PRINT_FEX_FIELD

  reg = p;
  ::signal( SIGINT, sigHandler );

  DmaStats d;
  RateMonitor<DmaStats> dstats(d);
  unsigned och0  =0;
  unsigned otot  =0;

  const QABase& base = *reinterpret_cast<QABase*>(reinterpret_cast<uint32_t*>(p->reg())+0x20000);
  { base.dump(); }

  PgpStats pgp(*p, channelMask);
  FexStats fexs(*p, channelMask, streamMask);
  CacheDump cache(*p);

  p->start();

  while(1) {
    usleep(1000000);

    printf("--------------\n");


    { unsigned uch0 = base.countAcquire;
      unsigned utot = base.countEnable;
      printf("eventCount: %08x:%08x [%d:%d]\n",uch0,utot,uch0-och0,utot-otot);
      och0 = uch0;
      otot = utot;
    }

    { DmaStats d(base);
      RateMonitor<DmaStats> dmaStats(d);
      dstats.dump(dmaStats);
      dstats = dmaStats; }

    pgp.dump();
    fexs.dump();
    cache.dump();
  }

  return 0;
}
