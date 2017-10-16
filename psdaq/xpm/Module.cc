#include "Module.hh"

#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <new>

using namespace Pds::Xpm;
using Pds::Cphw::Reg;

enum { MSG_CLEAR_FIFO =0, 
       MSG_DELAY_PWORD=1 };

static inline unsigned getf(unsigned i, unsigned n, unsigned sh)
{
  unsigned v = i;
  return (v>>sh)&((1<<n)-1);
}

static inline unsigned getf(const Pds::Cphw::Reg& i, unsigned n, unsigned sh)
{
  unsigned v = i;
  return (v>>sh)&((1<<n)-1);
}

static inline unsigned setf(Pds::Cphw::Reg& o, unsigned v, unsigned n, unsigned sh)
{
  unsigned r = unsigned(o);
  unsigned q = r;
  q &= ~(((1<<n)-1)<<sh);
  q |= (v&((1<<n)-1))<<sh;
  o = q;

  /*
  if (q != unsigned(o)) {
    printf("setf[%p] failed: %08x != %08x\n", &o, unsigned(o), q);
  }
  else if (q != r) {
    printf("setf[%p] passed: %08x [%08x]\n", &o, q, r);
  }
  */
  return q;
}

L0Stats::L0Stats() {
  l0Enabled=0; 
  l0Inhibited=0; 
  numl0=0; 
  numl0Inh=0; 
  numl0Acc=0; 
  memset(linkInh,0,32*sizeof(uint32_t)); 
  rx0Errs=0; 
}

LinkStatus::LinkStatus() {
  txReady = 0;
  rxReady = 0;
  isXpm   = 0;
  rxErrs  = 0;
}

void CoreCounts::dump() const
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
#undef PU64
}


CoreCounts Module::counts() const
{
  CoreCounts c;
  c.rxClkCount       = _timing.RxRecClks;
  c.txClkCount       = _timing.TxRefClks;
  c.rxRstCount       = _timing.RxRstDone;
  c.crcErrCount      = _timing.CRCerrors;
  c.rxDecErrCount    = _timing.RxDecErrs;
  c.rxDspErrCount    = _timing.RxDspErrs;
  c.bypassResetCount = (_timing.BuffByCnts >> 16) & 0xffff;
  c.bypassDoneCount  = (_timing.BuffByCnts >>  0) & 0xffff;
  c.rxLinkUp         = (_timing.CSR >> 1) & 0x01;
  c.fidCount         = _timing.Msgcounts;
  c.sofCount         = _timing.SOFcounts;
  c.eofCount         = _timing.EOFcounts;

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
                 linkInh[i+0],linkInh[i+1],linkInh[i+2],linkInh[i+3]);
#undef PU64
        printf("%9.9s: %x\n","rxErrs",rx0Errs);
}


Module* Module::locate()
{
  return new((void*)0) Module;
}

Module::Module()
{ /*init();*/ }

void Module::init()
{
  printf("Module:    paddr %x\n", unsigned(_paddr));

  printf("Index:     partition %u  link %u  linkDebug %u  amc %u  inhibit %u  tagStream %u\n",
         getf(_index,4,0),
         getf(_index,6,4),
         getf(_index,4,10),
         getf(_index,1,16),
         getf(_index,2,20),
         getf(_index,1,24));

  unsigned il = getLink();

  printf("DsLnkCfg:  %4.4s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s\n",
         "Link", "TxDelay", "Partn", "TrigSrc", "Loopback", "TxReset", "RxReset", "Enable");
  for(unsigned i=0; i<NDSLinks; i++) {
    setLink(i);
    printf("           %4u %8u %8u %8u %8u %8u %8u %8u\n",
           i,
           getf(_dsLinkConfig,20,0),
           getf(_dsLinkConfig,4,20),
           getf(_dsLinkConfig,4,24),
           getf(_dsLinkConfig,1,28),
           getf(_dsLinkConfig,1,29),
           getf(_dsLinkConfig,1,30),
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

void Module::linkTxDelay(unsigned link, unsigned v)
{
  setLink(link);
  setf(_dsLinkConfig, v, 20, 0);
}
unsigned Module::linkTxDelay(unsigned link) const
{
  setLink(link);
  return getf(_dsLinkConfig,    20, 0);
}

void Module::linkPartition(unsigned link, unsigned v)
{
  setLink(link);
  setf(_dsLinkConfig, v, 4, 20);
}
unsigned Module::linkPartition(unsigned link) const
{
  setLink(link);
  return getf(_dsLinkConfig, 4, 20);
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

L0Stats Module::l0Stats() const
{
  //  Lock the counters
  const_cast<Module&>(*this).lockL0Stats(true);
  L0Stats s;
  s.l0Enabled   = _l0Enabled;
  s.l0Inhibited = _l0Inhibited;
  s.numl0       = _numl0;
  s.numl0Inh    = _numl0Inh;
  s.numl0Acc    = _numl0Acc;
  //  for(unsigned i=0; i<NDSLinks; i++) {
  for(unsigned i=0; i<32; i++) {
    setLink(i);
    // if (getf(_dsLinkConfig,1,31))
      s.linkInh[i] = _inhibitCounts[i];
  }
  //  Release the counters
  const_cast<Module&>(*this).lockL0Stats(false);

  s.rx0Errs = rxLinkErrs(0);

  return s;
}

LinkStatus Module::linkStatus(unsigned link) const
{
  LinkStatus s;
  setLink(link);
  unsigned dsLinkStatus = _dsLinkStatus;
  //  printf("LinkStatus[%u] %08x\n", link,dsLinkStatus);
  s.txReady = getf(dsLinkStatus,1,17);
  s.rxReady = getf(dsLinkStatus,1,19);
  s.isXpm   = getf(dsLinkStatus,1,20);
  s.rxErrs  = getf(dsLinkStatus,16,0);
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

void Module::setL0Enabled(bool v)
{
  printf("L0Select: %08x\n", unsigned(_l0Select));
  setf(_l0Control,v?1:0,1,16);
}

bool Module::getL0Enabled() const
{
  return getf(_l0Control,1,16) ? 1 : 0;
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

void Module::pllBwSel(unsigned idx, unsigned sel)
{
  setAmc(idx);
  setf(_pllConfig,sel,4,0);
}
unsigned Module::pllBwSel(unsigned idx) const
{
  setAmc(idx);
  return getf(_pllConfig,4,0);
}

void Module::pllFrqTbl(unsigned idx, unsigned sel)
{
  setAmc(idx);
  setf(_pllConfig,sel,2,4);
}
unsigned Module::pllFrqTbl(unsigned idx) const
{
  setAmc(idx);
  return getf(_pllConfig,2,4);
}

void Module::pllFrqSel(unsigned idx, unsigned sel)
{
  setAmc(idx);
  setf(_pllConfig,sel,8,8);
}
unsigned Module::pllFrqSel(unsigned idx) const
{
  setAmc(idx);
  return getf(_pllConfig,8,8);
}

void Module::pllRateSel(unsigned idx, unsigned sel)
{
  setAmc(idx);
  setf(_pllConfig,sel,4,16);
}
unsigned Module::pllRateSel(unsigned idx) const
{
  setAmc(idx);
  return getf(_pllConfig,4,16);
}

void Module::pllPhsInc(unsigned idx)
{
  unsigned v = _pllConfig;
  setAmc(idx);
  _pllConfig = v|(1<<20);
  usleep(10);
  _pllConfig = v;
  usleep(10);
}

void Module::pllPhsDec(unsigned idx)
{
  unsigned v = _pllConfig;
  setAmc(idx);
  _pllConfig = v|(1<<21);
  usleep(10);
  _pllConfig = v;
  usleep(10);
}

void Module::pllBypass(unsigned idx, bool v)
{
  setAmc(idx);
  setf(_pllConfig,v?1:0,1,22);
}
bool Module::pllBypass(unsigned idx) const
{
  setAmc(idx);
  return getf(_pllConfig,1,22);
}

void Module::pllReset(unsigned idx)
{
  setAmc(idx);
  setf(_pllConfig,0,1,23);
  usleep(10);
  setf(_pllConfig,1,1,23);
}

unsigned Module::pllStatus0(unsigned idx) const
{
  setAmc(idx);
  return getf(_pllConfig,1,27);
}

unsigned Module::pllCount0(unsigned idx) const
{
  setAmc(idx);
  return getf(_pllConfig,3,24);
}

unsigned Module::pllStatus1(unsigned idx) const
{
  setAmc(idx);
  return getf(_pllConfig,1,31);
}

unsigned Module::pllCount1(unsigned idx) const
{
  setAmc(idx);
  return getf(_pllConfig,3,28);
}

void Module::pllSkew(unsigned idx, int skewSteps)
{
  unsigned v = _pllConfig;
  setAmc(idx);
  while(skewSteps > 0) {
    _pllConfig = v|(1<<20);
    usleep(10);
    _pllConfig = v;
    usleep(10);
    skewSteps--;
  }
  while(skewSteps < 0) {
    _pllConfig = v|(1<<21);
    usleep(10);
    _pllConfig = v;
    usleep(10);
    skewSteps++;
  }
}

void Module::dumpPll(unsigned idx) const
{
  setAmc(idx);
  printf("AMC[%d] pllConfig 0x%08x\n", idx, unsigned(_pllConfig));

  unsigned bwSel = getf(_pllConfig,4,0);
  unsigned frTbl = getf(_pllConfig,2,4);
  unsigned frSel = getf(_pllConfig,8,8);
  unsigned rate  = getf(_pllConfig,4,16);
  unsigned cnt0  = getf(_pllConfig,3,24);
  unsigned stat0 = getf(_pllConfig,1,27);
  unsigned cnt1  = getf(_pllConfig,3,28);
  unsigned stat1 = getf(_pllConfig,1,31);

  static const char lmh[] = {'L', 'H', 'M', 'm'};
  printf("  ");
  printf("FrqTbl %c  ", lmh[frTbl]);
  printf("FrqSel %c%c%c%c  ", lmh[getf(frSel,2,6)], lmh[getf(frSel,2,4)],
                             lmh[getf(frSel,2,2)], lmh[getf(frSel,2,0)]);
  printf("BwSel %c%c  ", lmh[getf(bwSel,2,2)], lmh[getf(bwSel,2,0)]);
  printf("Rate %c%c  ", lmh[getf(rate,2,2)], lmh[getf(rate,2,0)]);
  printf("Cnt0 %u  ", cnt0);
  printf("LOS %c  ", stat0 ? 'Y' : 'N');
  printf("Cnt1 %u  ", cnt1);
  printf("LOL %c  ", stat1 ? 'Y' : 'N');
  printf("\n");
}

void Module::dumpTiming(unsigned b) const
{
  Module& cthis = *const_cast<Module*>(this);
  Pds::Cphw::RingBuffer& ring = b==0 ? cthis._timing.ring0 : cthis._timing.ring1;

  ring.enable(false);
  ring.clear();
  ring.enable(true);
  usleep(10);
  ring.enable(false);
  ring.dump();
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

void     Module::setLinkDebug(unsigned v) const
{
  setf(const_cast<Pds::Cphw::Reg&>(_index),v,4,10);
}
unsigned Module::getLinkDebug() const     { return getf(_index,  4,10); }

void     Module::setAmc(unsigned v) const
{
  setf(const_cast<Pds::Cphw::Reg&>(_index),v,1,16);
}
unsigned Module::getAmc() const     { return getf(_index,  1,16); }

void     Module::setInhibit(unsigned v) {        setf(_index,v,2,20); }
unsigned Module::getInhibit() const     { return getf(_index,  2,20); }

void     Module::setTagStream(unsigned v) {        setf(_index,v,1,24); }
unsigned Module::getTagStream() const     { return getf(_index,  1,24); }

//  Update pipeline depth
//  Insert message to communicate downstream
void Module::setL0Delay(unsigned partition, unsigned v)
{
  setPartition(partition);
  setf(_pipelineDepth, v, 32, 0);
  setf(_messagePayload, v, 32, 0);
  setf(_message, ((1<<15) | MSG_DELAY_PWORD), 16, 0);
}

unsigned Module::getL0Delay(unsigned partition) const
{
  setPartition(partition);
  return getf(_pipelineDepth, 32, 0);
}

void Module::setL1TrgClr(unsigned partition, unsigned v)
{
  setPartition(partition);
  setf(_l1config0, v, 1, 0);
}
unsigned Module::getL1TrgClr(unsigned partition) const
{
  setPartition(partition);
  return getf(_l1config0, 1, 0);
}
void Module::setL1TrgEnb(unsigned partition, unsigned v)
{
  setPartition(partition);
  setf(_l1config0, v, 1, 16);
}
unsigned Module::getL1TrgEnb(unsigned partition) const
{
  setPartition(partition);
  return getf(_l1config0, 1, 16);
}
void Module::setL1TrgSrc(unsigned partition, unsigned v)
{
  setPartition(partition);
  setf(_l1config1, v, 4, 0);
}
unsigned Module::getL1TrgSrc(unsigned partition) const
{
  setPartition(partition);
  return getf(_l1config1, 4, 0);
}
void Module::setL1TrgWord(unsigned partition, unsigned v)
{
  setPartition(partition);
  setf(_l1config1, v, 9, 4);
}
unsigned Module::getL1TrgWord(unsigned partition) const
{
  setPartition(partition);
  return getf(_l1config1, 9, 4);
}
void Module::setL1TrgWrite(unsigned partition, unsigned v)
{
  setPartition(partition);
  setf(_l1config1, v, 1, 16);
}
unsigned Module::getL1TrgWrite(unsigned partition) const
{
  setPartition(partition);
  return getf(_l1config1, 1, 16);
}
void Module::messageHdr(unsigned partition, unsigned v)
{
  v |= (1<<15);
  setPartition(partition);
  setf(_message, v, 16, 0);
}
unsigned Module::messageHdr(unsigned partition) const
{
  setPartition(partition);
  return getf(_message, 16, 0);
}
void Module::messagePayload(unsigned partition, unsigned v)
{
  setPartition(partition);
  setf(_messagePayload, v, 32, 0);
}
unsigned Module::messagePayload(unsigned partition) const
{
  setPartition(partition);
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
