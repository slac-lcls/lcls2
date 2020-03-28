#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>
#include <semaphore.h>

#include "psdaq/epicstools/PVCached.hh"
using Pds_Epics::EpicsPVA;
using Pds_Epics::PVCached;

#include <string>
#include <vector>
#include <sstream>

static const double CLK_FREQ = 1300e6/7.;
static bool _dump=false;


namespace Pds {
  namespace Tpr {
    class RxDesc {
    public:
      RxDesc(uint32_t* d, unsigned sz) : maxSize(sz), data(d) {}
    public:
      uint32_t  maxSize;
      uint32_t* data;
    };
  };
};

#include "psdaq/tpr/Module.hh"
using namespace Pds::Tpr;

extern int optind;

void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -r <a..z>\n");
  printf("         -n <PV base name>\n");
  printf("         -d (dump)\n");
}

namespace Pds {
  namespace Tpr {
    class Channel {
    public:
      Channel() : bsaWid(0) {}
      unsigned             channel;
      TprBase::Destination destn;
      TprBase::FixedRate   rate;
      unsigned             bsaPre;
      unsigned             bsaDel;
      unsigned             bsaWid;
    };
  };
};

static const char* to_name(const std::string& base,
                           const char*        ext)
{
  std::string o(base);
  o += ":";
  o += std::string(ext);
  printf("to_name [%s]\n",o.c_str());
  return o.c_str();
}

static std::string to_name(const char* base,
                           unsigned    arg)
{
  std::ostringstream o;
  o << base << ":CH" << arg;
  printf("to_name [%s]\n",o.str().c_str());
  return o.str();
}

//
//  Generate triggers
//  Generate BSA
//
namespace Pds {
  class ChannelControl : public Pds_Epics::PVMonitorCb {
  public:
    ChannelControl(Tpr::TprBase& base,
                   std::string   name,
                   unsigned      chan) :
      _base      (base),
      _channel   (chan),
      _events    (base.channel[chan].evtCount),
      _mode      (to_name(name,"MODE" ),this),
      _delay     (to_name(name,"DELAY"),this),
      _width     (to_name(name,"WIDTH"),this),
      _polarity  (to_name(name,"POL")  ,this),
      _dstsel    (to_name(name,"DSTSEL"),this),
      _destns    (to_name(name,"DESTNS"),this),
      _rateSel   (to_name(name,"RSEL" ),this),
      _fixedRate (to_name(name,"FRATE"),this),
      _acRate    (to_name(name,"ARATE"),this),
      _acTimeslot(to_name(name,"ATS"  ),this),
      _seqIdx    (to_name(name,"SEQIDX"),this),
      _seqBit    (to_name(name,"SEQBIT"),this),
      _xPart     (to_name(name,"XPART"),this),
      _bsaStart  (to_name(name,"BSTART"),this),
      _bsaWidth  (to_name(name,"BWIDTH"),this),
      _rate      (to_name(name,"RATE"))
    {}
  public:
#define TOU(p)    p.getScalarAs<unsigned>()
#define TOI(p)    p.getScalarAs<int>()
#define STOU(p,s) unsigned(p.getScalarAs<double>()*s)
    void updated() {
      unsigned umode     = TOU(_mode);
      unsigned upolarity = TOU(_polarity);
      double   ddelay    = (_delay.getScalarAs<double>())*CLK_FREQ;
      unsigned udelay    = unsigned(ddelay);
      unsigned udelayTap = unsigned((ddelay-double(udelay))*63.);
      unsigned uwidth    = STOU(_width,CLK_FREQ);

      _base.setupTrigger( _channel,  // trigger output
                          _channel,  // event select channel
                          upolarity,
                          udelay,
                          uwidth,
                          udelayTap );

      unsigned ucontrol = _base.channel[_channel].control;
      _base.channel[_channel].control = 0;

      unsigned usel = TOU(_rateSel);
      unsigned ufr  = TOU(_fixedRate);
      unsigned uac  = TOU(_acRate);
      unsigned uts  = TOU(_acTimeslot);
      unsigned usn  = TOU(_seqIdx);
      unsigned usb  = TOU(_seqBit);
      unsigned uxp  = TOU(_xPart);
      unsigned urate;

      switch(usel) {
      case 0:
        urate  = ufr&0xf;
        break;
      case 1:
        urate  = (1<<11) | ((uac&0x7)<<0) | ((uts&0x3f)<<3);
        break;
      case 2:
        urate  = (2<<11) | ((usb&0x0f)<<0) | ((usn&0x1f)<<4);
        break;
      case 3:
        urate  = (3<<11) | (uxp&0xf);
        break;
      default:
        return;
      }

      unsigned destsel;
      unsigned udest = TOU(_dstsel);
      switch(udest) {
      case 0: // DontCare
        destsel = (2<<16);
        break;
      case 1: // Exclude
        destsel = (1<<16);
        break;
      default: // Include
        destsel = (0<<16);
        break;
      }

      destsel |= TOU(_destns)&0xffff;
      _base.channel[_channel].evtSel  = (destsel<<13) | (urate<<0);

      int bsaStart = TOI(_bsaStart);
      unsigned ubsaPS = bsaStart<0 ? -bsaStart : 0;
      unsigned ubsaDL = bsaStart<0 ? 0 : bsaStart;
      _base.channel[_channel].bsaDelay = (ubsaPS<<20) | (ubsaDL<<0);

      unsigned ubsaW = TOU(_bsaWidth);

      //
      //  Note that the kernel module enables/disables the DMA bit
      //  whenever an application opens the device.
      //
      //      unsigned u = 0;
      unsigned u = ucontrol & ~0x3;
      switch(umode) {
      case 0: // Disable
        ubsaW=0;
        break;
      case 1: // Trigger
        u |= 0x1;
        ubsaW=0;
        break;
      case 2: // +Readout
        u |= 0x1;
        ubsaW=0;
        break;
      case 3: // +BSA
      default:
        u |= 0x3;
        break;
      }

      _base.channel[_channel].bsaWidth = ubsaW;
      _base.channel[_channel].control = u;

      printf("Chan %u [%p]:  control %x [%x]  evtsel %x [%x]\n",
             _channel, &_base.channel[_channel],
             u, _base.channel[_channel].control,
             _base.channel[_channel].evtSel,
             (destsel<<13) | (urate<<0) );
    }
    void report(double dt) {
      unsigned events = _base.channel[_channel].evtCount;
      //      double rate = double(events-_events)/dt;
      double rate = double(events);
      _rate.putC(rate);
      _events=events;
    }
  private:
    Tpr::TprBase& _base;
    unsigned      _channel;
    unsigned      _events;
    EpicsPVA      _mode;
    EpicsPVA      _delay;
    EpicsPVA      _width;
    EpicsPVA      _polarity;
    EpicsPVA      _dstsel;
    EpicsPVA      _destns;
    EpicsPVA      _rateSel;
    EpicsPVA      _fixedRate;
    EpicsPVA      _acRate;
    EpicsPVA      _acTimeslot;
    EpicsPVA      _seqIdx;
    EpicsPVA      _seqBit;
    EpicsPVA      _xPart;
    EpicsPVA      _bsaStart;
    EpicsPVA      _bsaWidth;
    PVCached      _rate;
  };

#define PVPUT( pv, value ) {                            \
    pv.putC(double(value));                             \
  }
#define PVPUTD( vreg, ovalue, pv ) {                    \
    unsigned nvalue = vreg;                             \
    double value = double(nvalue-ovalue)/dt;            \
    pv.putC(value);                                     \
    ovalue = nvalue;                                    \
  }

  class TprControl : public Pds_Epics::PVMonitorCb {
  public:
    TprControl(TprReg* p, const char* name) :
      _dev       ( p ),
      _CSR       ( p->tpr.CSR ),
      _frames    ( p->tpr.Msgcounts ),
      _rxClks    ( p->tpr.RxRecClks ),
      _txClks    ( p->tpr.TxRefClks ),
      _accSelect ( to_name(name,"ACCSEL"   ), this ),
      _linkState ( to_name(name,"LINKSTATE") ),
      _linkLatch ( to_name(name,"LINKLATCH") ),
      _rxErrs    ( to_name(name,"RXERRS"   ) ),
      _rxErrsRst ( to_name(name,"RXERRSCL" ), this ),
      _vsnErr    ( to_name(name,"VSNERR"   ) ),
      _frameRate ( to_name(name,"FRAMERATE") ),
      _rxClkRate ( to_name(name,"RXCLKRATE") ),
      _txClkRate ( to_name(name,"TXCLKRATE") ),
      _frameVsn  ( to_name(name,"FRAMEVSN" ) ),
      _rxPolarity( to_name(name,"RXPOL"    ), this ),
      _irqEna    ( to_name(name,"IRQENA"   ) ),
      _evtCnt    ( to_name(name,"EVTCNT"   ) )
    {
      printf("FpgaVersion: %08x\n", p->version.FpgaVersion);
      printf("BuildStamp: %s\n", p->version.buildStamp().c_str());

      p->xbar.dump();
      p->tpr.dump();
      p->base.dump();
      p->dma.dump();

      for(unsigned i=0; i<TprBase::NCHANNELS; i++)
        _channels.push_back(new ChannelControl(p->base, to_name(name,i), i));

      PVPUT( _linkState, (_CSR&(1<<1)?1:0) );
      PVPUT( _linkLatch, (_CSR&(1<<5)?1:0) );

      timespec tvo;
      clock_gettime(CLOCK_REALTIME,&tvo);
      while(1) {
        usleep(1000000);
        timespec tv;
        clock_gettime(CLOCK_REALTIME,&tv);
        double dt = double(tv.tv_sec-tvo.tv_sec)+
          1.e-9*(double(tv.tv_nsec)-double(tvo.tv_nsec));
        tvo = tv;
        report(dt);
        for(unsigned i=0; i<TprBase::NCHANNELS; i++)
          _channels[i]->report(dt);

        if (_dump) {
          _dev->ring1.enable(false);
          _dev->ring1.dump();
          _dev->ring1.clear();
          _dev->ring1.enable(true);
        }
      }

      //  Enable interrupts (seems to be stuck)
      _dev->csr.irqEnable = 1;
    }
    ~TprControl() {}
  public:
    void updated() {
      unsigned upol = TOU(_rxPolarity);
      _dev->tpr.rxPolarity(upol);
      unsigned uacc = TOU(_accSelect);
      _dev->tpr .clkSel(uacc!=0);
      unsigned urst = TOU(_rxErrsRst);
      if (urst)
        _dev->tpr.resetCounts();
    }
    void report(double dt) {
      { unsigned v = _dev->tpr.CSR;
        //        unsigned v_delta = v ^ _CSR;
        PVPUT( _linkState, (v&(1<<1)?1:0) );
        PVPUT( _linkLatch, (v&(1<<5)?1:0) );
        _CSR = v;
      }

      { unsigned v = _dev->tpr.CRCerrors + _dev->tpr.RxDspErrs + _dev->tpr.RxDecErrs;
        PVPUT( _rxErrs, v ); }

      PVPUT( _vsnErr  , _dev->tpr.vsnErr() );
      PVPUT( _frameVsn,  _dev->tpr.FrameVersion );

      PVPUTD( _dev->tpr.Msgcounts, _frames, _frameRate );

      { unsigned v = _dev->tpr.RxRecClks;
        double value = 16.e-6*double(v-_rxClks)/dt;
        _rxClkRate.putFrom<double>(value);
        _rxClks = v; }
      { unsigned v = _dev->tpr.TxRefClks;
        double value = 16.e-6*double(v-_txClks)/dt;
        _txClkRate.putFrom<double>(value);
        _txClks = v; }

      PVPUT( _irqEna   , _dev->csr.irqEnable  );
      PVPUT( _evtCnt   , _dev->csr.trigMaster ); 
    }
  private:
    TprReg*  _dev;
    unsigned _CSR;
    unsigned _frames;
    unsigned _rxClks;
    unsigned _txClks;
    EpicsPVA _accSelect;
    PVCached _linkState;
    PVCached _linkLatch;
    PVCached _rxErrs;
    EpicsPVA _rxErrsRst;
    PVCached _vsnErr;
    PVCached _frameRate;
    PVCached _rxClkRate;
    PVCached _txClkRate;
    PVCached _frameVsn;
    EpicsPVA _rxPolarity;
    PVCached _irqEna;
    PVCached _evtCnt;
    std::vector<ChannelControl*> _channels;
  };
};



int main(int argc, char** argv) {

  extern char* optarg;
  char tprid='a';
  const char* name = "TPR";

  //  char* endptr;
  int c;
  bool lUsage = false;
  while ( (c=getopt( argc, argv, "r:n:dh?")) != EOF ) {
    switch(c) {
    case 'r':
      tprid  = optarg[0];
      if (strlen(optarg) != 1) {
        printf("%s: option `-r' parsing error\n", argv[0]);
        lUsage = true;
      }
      break;
    case 'n':
      name = optarg;
      break;
    case 'd':
      _dump = true;
      break;
    case 'h':
      usage(argv[0]);
      exit(0);
    case '?':
    default:
      lUsage = true;
      break;
    }
  }

  if (optind < argc) {
    printf("%s: invalid argument -- %s\n",argv[0], argv[optind]);
    lUsage = true;
  }

  if (lUsage) {
    usage(argv[0]);
    exit(1);
  }

  {
    char dev[16];
    sprintf(dev,"/dev/tpr%c",tprid);
    printf("Using tpr %s\n",dev);

    int fd = open(dev, O_RDWR);
    if (fd<0) {
      perror("Could not open");
      return -1;
    }

    { TprReg* p = new(0)TprReg;
      printf("AxiVersion: %p\n", &p->version);
      printf("XBar      : %p\n", &p->xbar);
      printf("TprCsr    : %p\n", &p->csr);
      printf("DmaControl: %p\n", &p->dma);
      printf("TprBase   : %p\n", &p->base);
      printf("TprCore   : %p\n", &p->tpr);
      printf("Ring0     : %p\n", &p->ring0);
    }

    void* ptr = mmap(0, sizeof(TprReg), PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
      perror("Failed to map");
      return -2;
    }

    reinterpret_cast<TprReg*>(ptr)->xbar.outMap[2]=0;
    reinterpret_cast<TprReg*>(ptr)->xbar.outMap[3]=1;

    { RingB& ring = reinterpret_cast<TprReg*>(ptr)->ring0;
      ring.clear();
      ring.enable(true);
      usleep(1000);
      ring.enable(false);
      ring.dumpFrames(); }
    { RingB& ring = reinterpret_cast<TprReg*>(ptr)->ring1;
      ring.clear();
      ring.enable(true);
      usleep(1000);
      ring.enable(false);
      ring.dump(); }

    Pds::TprControl* tpr = new Pds::TprControl(reinterpret_cast<TprReg*>(ptr),
                                               name);
    delete tpr;
  }

  return 0;
}
