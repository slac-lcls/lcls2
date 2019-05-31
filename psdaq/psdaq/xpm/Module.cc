#include "Module.hh"
#include "psdaq/cphw/Utils.hh"
#include "psdaq/xpm/XpmSequenceEngine.hh"

#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <new>

using namespace Pds::Xpm;
using Pds::Cphw::Reg;

enum { MSG_CLEAR_FIFO =0, 
       MSG_DELAY_PWORD=1 };

static unsigned _verbose = 0;
static unsigned _feature_rev = 0;

L0Stats::L0Stats() {
  l0Enabled=0; 
  l0Inhibited=0; 
  numl0=0; 
  numl0Inh=0; 
  numl0Acc=0; 
  memset(linkInhEv,0,32*sizeof(uint32_t)); 
  memset(linkInhTm,0,32*sizeof(uint32_t)); 
  rx0Errs=0; 
}

LinkStatus::LinkStatus() {
  txReady = 0;
  rxReady = 0;
  txResetDone = 0;
  rxResetDone = 0;
  isXpm   = 0;
  rxRcvs  = 0;
  rxErrs  = 0;
}

void TimingCounts::dump() const
{
#define PU64(title,stat) printf("%9.9s: %lx\n",#title,stat)
  PU64(rxClkCnt,  rxClkCount);
  PU64(txClkCnt,  txClkCount);
  PU64(rxRstCnt,  rxRstCount);
  PU64(crcErrs,   crcErrCount);
  PU64(rxDecErrs, rxDecErrCount);
  PU64(rxDspErrs, rxDspErrCount);
  PU64(bypRstCnt, bypassResetCount);
  PU64(bypDneCnt, bypassDoneCount);
  PU64(rxLinkUp,  rxLinkUp);
  PU64(fidCnt,    fidCount);
  PU64(sofCnt,    sofCount);
  PU64(eofCnt,    eofCount);
  //  PU64(rxAlign,   rxAlign);
#undef PU64
}

TimingCounts::TimingCounts(const Cphw::TimingRx&   timing,
                           const Cphw::GthRxAlign& align)
{
  rxClkCount       = timing.RxRecClks;
  txClkCount       = timing.TxRefClks;
  rxRstCount       = timing.RxRstDone;
  crcErrCount      = timing.CRCerrors;
  rxDecErrCount    = timing.RxDecErrs;
  rxDspErrCount    = timing.RxDspErrs;
  bypassResetCount = (timing.BuffByCnts >> 16) & 0xffff;
  bypassDoneCount  = (timing.BuffByCnts >>  0) & 0xffff;
  rxLinkUp         = (timing.CSR >> 1) & 0x01;
  fidCount         = timing.Msgcounts;
  sofCount         = timing.SOFcounts;
  eofCount         = timing.EOFcounts;
  rxAlign          = 0;
  //  rxAlign          = align.gthAlignLast & 0x7f;
}

CoreCounts Module::counts() const
{
  CoreCounts c;
  c.us = TimingCounts(_usTiming, _usGthAlign);
  c.cu = TimingCounts(_cuTiming, _cuGthAlign);
  return c;
}

void L0Stats::dump() const
{
#define PU64(title,stat) printf("%9.9s: %lx\n",#title,stat)
        PU64(Enabled  ,l0Enabled);
        PU64(Inhibited,l0Inhibited);
        PU64(L0,numl0);
        PU64(L0Inh,numl0Inh);
        PU64(L0Acc,numl0Acc);
        for(unsigned i=0; i<32; i+=4)
          printf("Inhibit[%u]: %08x %08x %08x %08x\n", i,
                 linkInhTm[i+0],linkInhTm[i+1],linkInhTm[i+2],linkInhTm[i+3]);
#undef PU64
        printf("%9.9s: %x\n","rxErrs",rx0Errs);
}


Module* Module::locate()
{
  return new((void*)0) Module;
}

unsigned Module::feature_rev() { return _feature_rev; }

Module::Module()
{ /*init();*/ }

void Module::init()
{
  std::string bld = _version.buildStamp();
  if (bld.find("xtpg")!=std::string::npos)
    _feature_rev = 1;

  printf("Module:    paddr %x @ %p\n", unsigned(_paddr), &_paddr);

  printf("Index:     partition %u  link %u  linkDebug %u  amc %u  inhibit %u  tagStream %u\n",
         getf(_index,4,0),
         getf(_index,6,4),
         getf(_index,4,10),
         getf(_index,1,16),
         getf(_index,2,20),
         getf(_index,1,24));

  unsigned il = getLink();

  printf("DsLnkCfg:  %4.4s %9.9s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s\n",
         "Link", "GroupMask", "RxTmo", "TrigSrc", "Loopback", "TxReset", "RxReset", "TxPllRst", "RxPllRst", "Enable");
  for(unsigned i=0; i<NDSLinks; i++) {
    setLink(i);
    printf("           %4u      0x%02x %8u %8u %8u %8u %8u %8u %8u %8u\n",
           i,
           getf(_dsLinkConfig,8,0),
           getf(_dsLinkConfig,9,9),
           getf(_dsLinkConfig,4,24),
           getf(_dsLinkConfig,1,28),
           getf(_dsLinkConfig,1,29),
           getf(_dsLinkConfig,1,30),
           getf(_dsLinkConfig,1,18),
           getf(_dsLinkConfig,1,19),
           getf(_dsLinkConfig,1,31));
  }

  printf("DsLnkStat: %4.4s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s\n",
         "Link", "RxErr", "TxRstDn", "TxRdy", "RxRstDn", "RxRdy", "isXpm");
  for(unsigned i=0; i<NDSLinks; i++) {
    setLink(i);
    printf("           %4u %8u %8u %8u %8u %8u %8u\n",
           i,
           getf(_dsLinkStatus,16,0),
           getf(_dsLinkStatus,1,16),
           getf(_dsLinkStatus,1,17),
           getf(_dsLinkStatus,1,18),
           getf(_dsLinkStatus,1,19),
           getf(_dsLinkStatus,1,20));
  }

  setLink(il);

  //  Set the timing crossbar
  _xbar.setOut( Pds::Cphw::XBar::BP  , Pds::Cphw::XBar::FPGA );
  _xbar.setOut( Pds::Cphw::XBar::RTM0, Pds::Cphw::XBar::FPGA );
  _xbar.setOut( Pds::Cphw::XBar::RTM1, Pds::Cphw::XBar::FPGA );

  _hsRepeater[0].init();
  _hsRepeater[1].init();
  _hsRepeater[3].init();
  _hsRepeater[4].init();

  //  Consider resetting the backplane tx link
  //  Would be disruptive to already running acquisitions
  //  txLinkReset(16);

  /*
  printf("l0 enabled [%x]  reset [%x]\n",
         unsigned(_l0Control)>>16, unsigned(_l0Control)&0xffff);
  printf("l0 rate [%x]  dest [%x]\n",
         unsigned(_l0Select)&0xffff, unsigned(_l0Select)>>16);
  printf("link  fullEn [%x]  loopEn [%x]\n",
         unsigned(_dsLinkEnable)&0xffff,
         unsigned(_dsLinkEnable)>>16);
  printf("txReset PMA [%x]  All [%x]\n",
         unsigned(_dsTxLinkStatus)&0xffff,
         unsigned(_dsTxLinkStatus)>>16);
  printf("rxReset PMA [%x]  All [%x]\n",
         unsigned(_dsRxLinkStatus)&0xffff,
         unsigned(_dsRxLinkStatus)>>16);
  printf("l0Enabld %x\n", unsigned(_l0Enabled));
  printf("l0Inh    %x\n", unsigned(_l0Inhibited));
  printf("numl0    %x\n", unsigned(_numl0));
  printf("numl0Inh %x\n", unsigned(_numl0Inh));
  printf("numl0Acc %x\n", unsigned(_numl0Acc));
  */
}

void Module::clearLinks()
{
  for(unsigned i=0; i<NDSLinks; i++) {
    setLink(i);
    setf(_dsLinkConfig,0,1,31);
  }
}

void Module::linkRxTimeOut(unsigned link, unsigned v)
{
  setLink(link);
  setf(_dsLinkConfig, v, 9, 9);
}
unsigned Module::linkRxTimeOut(unsigned link) const
{
  setLink(link);
  return getf(_dsLinkConfig,    9, 9);
}

void Module::linkGroupMask(unsigned link, unsigned v)
{
  setLink(link);
  setf(_dsLinkConfig, v, 8, 0);
}
unsigned Module::linkGroupMask(unsigned link) const
{
  setLink(link);
  return getf(_dsLinkConfig, 8, 0);
}

void Module::linkTrgSrc(unsigned link, unsigned v)
{
  setLink(link);
  setf(_dsLinkConfig, v, 4, 24);
}
unsigned Module::linkTrgSrc(unsigned link) const
{
  setLink(link);
  return getf(_dsLinkConfig, 4, 24);
}

void Module::linkLoopback(unsigned link, bool v)
{
  setLink(link);
  setf(_dsLinkConfig,v?1:0,1,28);
}
bool Module::linkLoopback(unsigned link) const
{
  setLink(link);
  return getf(_dsLinkConfig,1,28);
}

void Module::txLinkReset(unsigned link)
{
  setLink(link);
  setf(_dsLinkConfig,1,1,29);
  usleep(10);
  setf(_dsLinkConfig,0,1,29);
}

void Module::rxLinkReset(unsigned link)
{
  setLink(link);
  setf(_dsLinkConfig,1,1,30);
  usleep(10);
  setf(_dsLinkConfig,0,1,30);
}

void Module::txLinkPllReset(unsigned link)
{
  setLink(link);
  setf(_dsLinkConfig,1,1,18);
  usleep(10);
  setf(_dsLinkConfig,0,1,18);
}

void Module::rxLinkPllReset(unsigned link)
{
  setLink(link);
  setf(_dsLinkConfig,1,1,19);
  usleep(10);
  setf(_dsLinkConfig,0,1,19);
}

void Module::rxLinkDump(unsigned link) const
{
  Module& cthis = *const_cast<Module*>(this);
  cthis.setRingBChan(link);

  Pds::Cphw::RingBuffer& ring = cthis._rxRing;
  ring.clear();
  ring.enable(true);
  usleep(100);
  ring.enable(false);
  ring.dump(20);
}

void Module::linkEnable(unsigned link, bool v)
{
  setLink(link);
  //usleep(10);
  //unsigned q = _dsLinkConfig;
  setf(_dsLinkConfig,v?1:0,1,31);
  //unsigned r = _dsLinkConfig;
  //printf("linkEnable[%u,%c] %x -> %x\n", link,v?'T':'F',q,r);
}

bool Module::linkEnable(unsigned link) const
{
  setLink(link);
  return getf(_dsLinkConfig,1,31);
}

bool     Module::linkRxReady (unsigned link) const
{
  setLink(link);
  return getf(_dsLinkStatus,1,19);
}

bool     Module::linkTxReady (unsigned link) const
{
  setLink(link);
  return getf(_dsLinkStatus,1,17);
}

bool     Module::linkIsXpm   (unsigned link) const
{
  setLink(link);
  return getf(_dsLinkStatus,1,20);
}

bool     Module::linkRxErr   (unsigned link) const
{
  setLink(link);
  return getf(_dsLinkStatus,16,0)!=0;
}


bool Module::l0Enabled() const { return getf(_l0Control,1,16); }

L0Stats Module::l0Stats(bool master) const
{
  //  Lock the counters
  const_cast<Module&>(*this).lockL0Stats(true);
  L0Stats s;
  clock_gettime(CLOCK_REALTIME,&s.time);
  if (master) {
    s.l0Enabled   = _l0Enabled;
    s.l0Inhibited = _l0Inhibited;
    s.numl0       = _numl0;
    s.numl0Inh    = _numl0Inh;
    s.numl0Acc    = _numl0Acc;
    //  for(unsigned i=0; i<NDSLinks; i++) {
    for(unsigned i=0; i<32; i++) {
      setLink(i);
      // if (getf(_dsLinkConfig,1,31))
      s.linkInhEv[i] = _inhibitEvCounts[i];
      s.linkInhTm[i] = _inhibitTmCounts[i];
    }
  }
  else {
    for(unsigned i=0; i<32; i++) {
      setLink(i);
      s.linkInhTm[i] = _inhibitTmCounts[i];
    }
  }
  //  Release the counters
  const_cast<Module&>(*this).lockL0Stats(false);

  s.rx0Errs = rxLinkErrs(0);

  if (_verbose > 1)
    s.dump();

  return s;
}

LinkStatus Module::linkStatus(unsigned link) const
{
  LinkStatus s;
  setLink(link);
  unsigned dsLinkStatus = _dsLinkStatus;
  //  printf("LinkStatus[%u] %08x\n", link,dsLinkStatus);
  s.txResetDone = getf(dsLinkStatus,1,16);
  s.txReady     = getf(dsLinkStatus,1,17);
  s.rxResetDone = getf(dsLinkStatus,1,18);
  s.rxReady     = getf(dsLinkStatus,1,19);
  s.isXpm   = getf(dsLinkStatus,1,20);
  s.rxErrs  = getf(_dsLinkStatus,16,0);
  s.rxRcvs  = _dsLinkRcvs;
  s.remoteLinkId = _remoteLinkId;
  if (link == 16) {
    s.rxRcvs  = -1;
    s.rxErrs  = -1;
  }
  return s;
}

void Module::linkStatus(LinkStatus* links) const
{
  for(unsigned i=0; i<32; i++)
    links[i] = linkStatus(i);
}

unsigned Module::rxLinkErrs(unsigned link) const
{
  setLink(link);
  return getf(_dsLinkStatus,16,0);
}


unsigned Module::txLinkStat() const
{
  unsigned ls = 0;
  for(unsigned i=0; i<NDSLinks; i++) {
    setLink(i);
    ls |= getf(_dsLinkStatus,1,17) << i;
  }
  return ls;
}

unsigned Module::rxLinkStat() const
{
  unsigned ls = 0;
  for(unsigned i=0; i<NDSLinks; i++) {
    setLink(i);
    ls |= getf(_dsLinkStatus,1,19) << i;
  }
  return ls;
}
void Module::resetL0(bool v)
{
  setf(_l0Control,v?1:0,1,0);
}
void Module::resetL0()
{
  resetL0(true);
  usleep(1);
  resetL0(false);
}
bool Module::l0Reset() const
{
  return getf(_l0Control,1,0);
}

void Module::master      (bool v)
{
  setf(_l0Control,v?1:0,1,30);
}

bool Module::master      () const
{
  return getf(_l0Control,1,30)!=0;
}

void Module::setL0Enabled(bool v)
{
  if (_verbose>0) 
    printf("setL0Enabled: L0Select: %08x\n", unsigned(_l0Select));

  setf(_l0Control,v?1:0,1,16);
}

bool Module::getL0Enabled() const
{
  return getf(_l0Control,1,16)!=0;
}

void Module::groupL0Reset  (unsigned m) 
{
  setPartition(nlsb(m)); // hack to prevent overwrite
  _groupL0Reset   = m; 
}

void Module::groupL0Enable (unsigned m) 
{
  setPartition(nlsb(m));
  _groupL0Enable  = m; 
}

void Module::groupL0Disable(unsigned m) 
{
  setPartition(nlsb(m));
  _groupL0Disable = m; 
}

void Module::groupMsgInsert(unsigned m) 
{
  setPartition(nlsb(m));
  _groupMsgInsert = m; 
}

void Module::setL0Select_FixedRate(unsigned rate)
{
  unsigned rateSel = (0<<14) | (rate&0xf);
  unsigned destSel = _l0Select >> 16;
  _l0Select = (destSel<<16) | rateSel;
}

void Module::setL0Select_ACRate   (unsigned rate, unsigned tsmask)
{
  unsigned rateSel = (1<<14) | ((tsmask&0x3f)<<3) | (rate&0x7);
  unsigned destSel = _l0Select >> 16;
  _l0Select = (destSel<<16) | rateSel;
}

void Module::setL0Select_Sequence (unsigned seq , unsigned bit)
{
  unsigned rateSel = (2<<14) | ((seq&0x3f)<<8) | (bit&0xf);
  unsigned destSel = _l0Select >> 16;
  _l0Select = (destSel<<16) | rateSel;
}

void Module::setL0Select_Destn    (unsigned mode, unsigned mask)
{
  unsigned rateSel = _l0Select & 0xffff;
  unsigned destSel = (mode<<15) | (mask&0xf);
  _l0Select = (destSel<<16) | rateSel;
}

void Module::lockL0Stats(bool v)
{
  setf(_l0Control,v?0:1,1,31);
}

void Module::setRingBChan(unsigned chan)
{
  setf(_index,chan,4,10);
}

#define PLL_MOD(func)                                  \
  void Module::pll##func(unsigned idx, int val) {      \
    setAmc(idx);                                       \
    _amcPll.func(val); }

#define PLL_ACC(func)                                   \
  int  Module::pll##func(unsigned idx) const {          \
    setAmc(idx);                                        \
    return _amcPll.func(); }

PLL_MOD(BwSel)
PLL_ACC(BwSel)
PLL_MOD(FrqTbl)
PLL_ACC(FrqTbl)
PLL_MOD(FrqSel)
PLL_ACC(FrqSel)
PLL_MOD(RateSel)
PLL_ACC(RateSel)
PLL_MOD(Skew)

#undef PLL_MOD
#define PLL_MOD(func)                           \
  void Module::pll##func(unsigned idx) {        \
    setAmc(idx);                                \
    _amcPll.func(); }

PLL_MOD(PhsInc)
PLL_MOD(PhsDec)
PLL_MOD(Reset)

#undef PLL_ACC
#define PLL_ACC(func)                                   \
  int  Module::pll##func(unsigned idx) const {          \
    setAmc(idx);                                        \
    return _amcPll.func(); }

PLL_ACC(Status0)
PLL_ACC(Count0)
PLL_ACC(Status1)
PLL_ACC(Count1)

void Module::pllBypass(unsigned idx, bool v)
{
  setAmc(idx);
  _amcPll.Bypass(v);
}
bool Module::pllBypass(unsigned idx) const
{
  setAmc(idx);
  return _amcPll.Bypass();
}

PllStats Module::pllStat(unsigned idx) const
{
  setAmc(idx);
  PllStats s;
  s.lol      = _amcPll.Status0();
  s.los      = _amcPll.Status1();
  s.lolCount = _amcPll.Count0();
  s.losCount = _amcPll.Count1();
  return s;
}

void Module::dumpPll(unsigned idx) const
{
  setAmc(idx);
  printf("AMC[%d] pllConfig 0x%08x\n", idx, unsigned(_amcPll._config));
  _amcPll.dump();
}

void Module::dumpTiming(unsigned b) const
{
  printf("dumpTiming deprecated\n");
}

void Module::setVerbose(unsigned v) 
{
  _verbose = v;
}

void Module::setTimeStamp()
{
  struct tm tm_s;
  tm_s.tm_year = 1995;
  time_t t0 = mktime(&tm_s);
  timespec ts;
  clock_gettime(CLOCK_REALTIME,&ts);
  uint64_t t = (ts.tv_sec-t0);
  t <<= 32;
  t |= ts.tv_nsec;
  _timestamp = t;
}

void Module::setCuInput(unsigned v)
{
  if (_feature_rev>0) {
    printf("Xpm::Module::setCuInput %x\n",v);
    Pds::Cphw::XBar::Map q((Pds::Cphw::XBar::Map)v);
    _xbar.setOut( Pds::Cphw::XBar::FPGA, q);
    _xbar.setOut( Pds::Cphw::XBar::RTM0, q);
    _xbar.setOut( Pds::Cphw::XBar::RTM1, q);
  }
}

void Module::setCuDelay(unsigned v)
{
  if (_feature_rev>0)
    _cuDelay = v;
}

void Module::setCuBeamCode(unsigned v)
{
  if (_feature_rev>0)
    _cuBeamCode = v;
}

void Module::clearCuFiducialErr(unsigned v)
{
  if (_feature_rev>0 && v)
    _cuFiducialIntv = v;
}

void     Module::setPartition(unsigned v) const
{
  setf(const_cast<Pds::Cphw::Reg&>(_index),v,4,0);
}
unsigned Module::getPartition() const     { return getf(_index,  4,0); }

void Module::setLink(unsigned v) const
{
  setf(const_cast<Pds::Cphw::Reg&>(_index),v,6,4);
}
unsigned Module::getLink() const { return getf(_index,6,4); }

void     Module::setAmc(unsigned v) const
{
  setf(const_cast<Pds::Cphw::Reg&>(_index),v,1,16);
}
unsigned Module::getAmc() const     { return getf(_index,  1,16); }

void     Module::setInhibit(unsigned v) {        setf(_index,v,2,20); }
unsigned Module::getInhibit() const     { return getf(_index,  2,20); }

void     Module::setTagStream(unsigned v) {        setf(_index,v,1,24); }
unsigned Module::getTagStream() const     { return getf(_index,  1,24); }

void Module::setL0Delay(unsigned v)
{
  unsigned r = v*200;
  r &= 0xffff;
  r |= v<<16;
  //  Update pipeline depth
  setf(_pipelineDepth, r, 32, 0);
}

unsigned Module::getL0Delay() const
{
  unsigned r = _pipelineDepth;
  return r>>16;
}

void Module::setL1TrgClr(unsigned v)
{
  setf(_l1config0, v, 1, 0);
}
unsigned Module::getL1TrgClr() const
{
  return getf(_l1config0, 1, 0);
}
void Module::setL1TrgEnb(unsigned v)
{
  setf(_l1config0, v, 1, 16);
}
unsigned Module::getL1TrgEnb() const
{
  return getf(_l1config0, 1, 16);
}
void Module::setL1TrgSrc(unsigned v)
{
  setf(_l1config1, v, 4, 0);
}
unsigned Module::getL1TrgSrc() const
{
  return getf(_l1config1, 4, 0);
}
void Module::setL1TrgWord(unsigned v)
{
  setf(_l1config1, v, 9, 4);
}
unsigned Module::getL1TrgWord() const
{
  return getf(_l1config1, 9, 4);
}
void Module::setL1TrgWrite(unsigned v)
{
  setf(_l1config1, v, 1, 16);
}
unsigned Module::getL1TrgWrite() const
{
  return getf(_l1config1, 1, 16);
}
void Module::messageInsert()
{
  setf(_message, 1, 1, 15);
}
void Module::messageHdr(unsigned v)
{
  setf(_message, v, 8, 0);
}
unsigned Module::messageHdr() const
{
  return getf(_message, 16, 0);
}
void Module::messagePayload(unsigned v)
{
  setf(_messagePayload, v, 32, 0);
}
unsigned Module::messagePayload() const
{
  return getf(_messagePayload, 32, 0);
}
void Module::inhibitInt(unsigned inh, unsigned v)
{
  setf(_inhibitConfig[inh], v-1, 12, 0);
}
unsigned Module::inhibitInt(unsigned inh) const
{
  return getf(_inhibitConfig[inh], 12, 0)+1;
}
void Module::inhibitLim(unsigned inh, unsigned v)
{
  setf(_inhibitConfig[inh], v-1, 4, 12);
}
unsigned Module::inhibitLim(unsigned inh) const
{
  return getf(_inhibitConfig[inh], 4, 12)+1;
}
void Module::inhibitEnb(unsigned inh, unsigned v)
{
  setf(_inhibitConfig[inh], v, 1, 31);
}
unsigned Module::inhibitEnb(unsigned inh) const
{
  return getf(_inhibitConfig[inh], 1, 31);
}

XpmSequenceEngine& Module::sequenceEngine()
{
  return *new XpmSequenceEngine(&_reserved_engine[0], 0);
}
