#include "Module.hh"

#include <stdio.h>

using namespace Pds::Tpr;

std::string AxiVersion::buildStamp() const {
  uint32_t tmp[64];
  for(unsigned i=0; i<64; i++)
    tmp[i] = BuildStamp[i];
  return std::string(reinterpret_cast<const char*>(tmp));
}

void XBar::setEvr( InMode  m ) { outMap[2] = m==StraightIn  ? 0:2; }
void XBar::setEvr( OutMode m ) { outMap[0] = m==StraightOut ? 2:0; }
void XBar::setTpr( InMode  m ) { outMap[3] = m==StraightIn  ? 1:3; }
void XBar::setTpr( OutMode m ) { outMap[1] = m==StraightOut ? 3:1; }
void XBar::dump() const { for(unsigned i=0; i<4; i++) printf("Out[%d]: %d\n",i,outMap[i]); }

void TprCsr::dump() const {
  printf("irqEnable [%p]: %08x\n",&irqEnable,irqEnable);
  printf("irqStatus [%p]: %08x\n",&irqStatus,irqStatus);
  printf("gtxDebug  [%p]: %08x\n",&gtxDebug  ,gtxDebug);
  printf("trigSel   [%p]: %08x\n",&trigMaster,trigMaster);
}

void TprBase::dump() const {
  static const unsigned NChan=14;
  static const unsigned NTrig=12;
  printf("\nchannel0  [%p]\n",&channel[0].control);
#define CHAN_REG(reg) {                                                 \
    printf("%s: ",#reg);                                                \
    for(unsigned i=0; i<NChan; i++)    printf("%08x ",channel[i].reg);  \
    printf("\n"); }
  CHAN_REG(control);
  CHAN_REG(evtCount);
  CHAN_REG(bsaCount);
  CHAN_REG(evtSel);
  CHAN_REG(bsaDelay);
  CHAN_REG(bsaWidth);
  printf("\ntrigger0  [%p]\n",&trigger[0].control);
#define TRIG_REG(reg) {                                                 \
    printf("%s: ",#reg);                                                \
    for(unsigned i=0; i<NTrig; i++)    printf("%08x ",trigger[i].reg);  \
    printf("\n"); }
  TRIG_REG(control);
  TRIG_REG(delay);
  TRIG_REG(width);
  TRIG_REG(delayTap);
#undef TRIG_REG
}

void TprCsr::setupDma    (unsigned fullThr) {
  dmaFullThr = fullThr;
}

void TprBase::setupDaq    (unsigned i,
                           unsigned partition) {
  channel[i].evtSel   = (1<<30) | (3<<14) | partition; // 
  channel[i].control = 5;
}

void TprBase::setupChannel(unsigned i,
                           Destination d,
                           FixedRate   r,
                           unsigned    bsaPresample,
                           unsigned    bsaDelay,
                           unsigned    bsaWidth) {
  channel[i].control  = 0;
  channel[i].evtSel   = (1<<30) | unsigned(r); //
  channel[i].bsaDelay = (bsaPresample<<20) | bsaDelay;
  channel[i].bsaWidth = bsaWidth;
  channel[i].control  = bsaWidth ? 7 : 5;
}

void TprBase::setupChannel(unsigned i,
                           Destination d,
                           ACRate      r,
                           unsigned    timeSlotMask,
                           unsigned    bsaPresample,
                           unsigned    bsaDelay,
                           unsigned    bsaWidth) {
  channel[i].control  = 0;
  channel[i].evtSel   = (1<<30) | (1<<11) | ((timeSlotMask&0x7fe)<<2) | (unsigned(r)&0x7); //
  channel[i].bsaDelay = (bsaPresample<<20) | bsaDelay;
  channel[i].bsaWidth = bsaWidth;
  channel[i].control  = bsaWidth ? 7 : 5;
}

void TprBase::setupChannel(unsigned i,
                           EventCode   r,
                           unsigned    bsaPresample,
                           unsigned    bsaDelay,
                           unsigned    bsaWidth) {
  channel[i].control  = 0;
  channel[i].evtSel   = (1<<30) | (2<<11) | (unsigned(r)&0xff); //
  channel[i].bsaDelay = (bsaPresample<<20) | bsaDelay;
  channel[i].bsaWidth = bsaWidth;
  channel[i].control  = bsaWidth ? 7 : 5;
}

void TprBase::setupTrigger(unsigned i,
                           unsigned source,
                           unsigned polarity,
                           unsigned delay,
                           unsigned width,
                           unsigned delayTap) {
  trigger[i].control  = (polarity ? (1<<16):0);
  usleep(1);
  trigger[i].delay    = delay;
  trigger[i].width    = width;
  trigger[i].control  = (source&0xffff) | (polarity ? (1<<16):0) | (1<<31);
  trigger[i].delayTap = delayTap;
}

void DmaControl::dump() const {
  printf("DMA Control\n");
  printf("\trxFreeStat : %8x\n",rxFreeStat);
  printf("\trxMaxFrame : %8x\n",rxMaxFrame);
  printf("\trxFifoSize : %8x\n",rxFifoSize&0x3ff);
  printf("\trxEmptyThr : %8x\n",(rxFifoSize>>16)&0x3ff);
  printf("\trxCount    : %8x\n",rxCount);
  printf("\tlastDesc   : %8x\n",lastDesc);
}

void DmaControl::test() {
  printf("DMA Control test\n");
  volatile unsigned v1 = rxMaxFrame;
  rxMaxFrame = 0x80001000;
  volatile unsigned v2 = rxMaxFrame;
  printf("\trxMaxFrame : %8x [%8x] %8x\n",v1,0x80001000,v2);

  v1     = rxFreeStat;
  rxFree = 0xdeadbeef;
  v2     = rxFreeStat;
  printf("\trxFreeStat [%8x], rxFree [%8x], lastDesc[%8x], rxFreeStat[%8x]\n",
         v1, 0xdeadbeef, lastDesc, v2);
}

void DmaControl::setEmptyThr(unsigned v)
{
  volatile unsigned v1 = rxFifoSize;
  rxFifoSize = ((v&0x3ff)<<16) | (v1&0x3ff);
}

bool TprCore::clkSel    () const {
  uint32_t v = CSR;
  return v&(1<<4);
}

void TprCore::clkSel    (bool lcls2) {
  volatile uint32_t v = CSR;
  v = lcls2 ? (v|(1<<4)) : (v&~(1<<4));
  CSR = v;
}

bool TprCore::rxPolarity() const {
  uint32_t v = CSR;
  return v&(1<<2);
}

void TprCore::rxPolarity(bool p) {
  volatile uint32_t v = CSR;
  v = p ? (v|(1<<2)) : (v&~(1<<2));
  CSR = v;
  usleep(10);
  CSR = v|(1<<3);
  usleep(10);
  CSR = v&~(1<<3);
}

void TprCore::resetRx() {
  volatile uint32_t v = CSR;
  CSR = (v|(1<<3));
  usleep(10);
  CSR = (v&~(1<<3));
}

void TprCore::resetRxPll() {
  volatile uint32_t v = CSR;
  CSR = (v|(1<<7));
  usleep(10);
  CSR = (v&~(1<<7));
}

void TprCore::resetCounts() {
  volatile uint32_t v = CSR;
  CSR = (v|1);
  usleep(10);
  CSR = (v&~1);
}

bool TprCore::vsnErr() const {
  volatile uint32_t v = CSR;
  return v & (1<<8);
}

void TprCore::dump() const {
  printf("SOFcounts: %08x\n", SOFcounts);
  printf("EOFcounts: %08x\n", EOFcounts);
  printf("Msgcounts: %08x\n", Msgcounts);
  printf("CRCerrors: %08x\n", CRCerrors);
  printf("RxRecClks: %08x\n", RxRecClks);
  printf("RxRstDone: %08x\n", RxRstDone);
  printf("RxDecErrs: %08x\n", RxDecErrs);
  printf("RxDspErrs: %08x\n", RxDspErrs);
  printf("CSR      : %08x\n", CSR); 
  printf("RxDspErrs: %08x\n", RxDspErrs);
  printf("TxRefClks: %08x\n", TxRefClks);
  printf("BypDone  : %04x\n", (BypassCnts>> 0)&0xffff);
  printf("BypResets: %04x\n", (BypassCnts>>16)&0xffff);
}


void RingB::enable(bool l) {
  volatile uint32_t v = csr;
  csr = l ? (v|(1<<31)) : (v&~(1<<31));
}
void RingB::clear() {
  volatile uint32_t v = csr;
  csr = v|(1<<30);
  usleep(10);
  csr = v&~(1<<30);
}
void RingB::dump(const char* fmt) const
{
  char sfmt[16];
  sprintf(sfmt,"%s%%c",fmt);
  for(unsigned i=0; i<0x1ff; i++)
    printf(sfmt,data[i],(i&0xf)==0xf ? '\n':' ');
}
void RingB::dumpFrames() const
{
#define print_u16 {                             \
    volatile uint32_t v  = (data[j++]<<16);     \
    printf("%8x ",v);                           \
  }
#define print_u32 {                             \
    volatile uint32_t v  = (data[j++]<<16);     \
    v = (v>>16) | (data[j++]<<16);              \
    printf("%8x ",v);                           \
  }
#define print_u32be {                           \
    volatile uint32_t v  = (data[j++]&0xffff);  \
    v = (v<<16) | (data[j++]&0xffff);           \
    printf("%8x ",v);                           \
  }
#define print_u64 {                             \
    uint64_t v  = (uint64_t(data[j++])<<48);    \
    v = (v>>16) | (uint64_t(data[j++])<<48);    \
    v = (v>>16) | (uint64_t(data[j++])<<48);    \
    v = (v>>16) | (uint64_t(data[j++])<<48);    \
    printf("%16lx ",v);                         \
  }
  printf("%8.8s %16.16s %16.16s %8.8s %8.8s %16.16s %16.16s %16.16s %16.16s\n",
         "Version","PulseID","TimeStamp","Markers","BeamReq",
         "BsaInit","BsaActiv","BsaAvgD","BsaDone");
  unsigned i=0;
  while(i<0x1fff) {
    if (data[i]==0x1b5f7) {  // Start of frame
      if (i+80 >= 0x1fff)
        break;
      unsigned j=i+2;
      print_u16; // version
      print_u64; // pulse ID
      print_u64; // time stamp
      print_u32; // rates/timeslot
      print_u32; // beamreq
      j += 12;
      print_u64; // bsainit
      print_u64; // bsaactive
      print_u64; // bsaavgdone
      print_u64; // bsadone
      printf("\n");
      i += 80;
    }
    else
      i++;
  }
#undef print_u32
#undef print_u32be
#undef print_u64
}


void TpgMini::setBsa(unsigned rate,
                     unsigned ntoavg,
                     unsigned navg)
{
  BsaDef[0].l = (1<<31) | (rate&0xffff);
  BsaDef[0].h = (navg<<16) | (ntoavg&0xffff);
}

void TpgMini::dump() const
{
  printf("ClkSel:\t%08x\n",ClkSel);
  printf("BaseCntl:\t%08x\n",BaseCntl);
  printf("PulseIdU:\t%08x\n",PulseIdU);
  printf("PulseIdL:\t%08x\n",PulseIdL);
  printf("TStampU:\t%08x\n",TStampU);
  printf("TStampL:\t%08x\n",TStampL);
  for(unsigned i=0; i<10; i++)
    printf("FixedRate[%d]:\t%08x\n",i,FixedRate[i]);
  printf("HistoryCntl:\t%08x\n",HistoryCntl);
  printf("FwVersion:\t%08x\n",FwVersion);
  printf("Resources:\t%08x\n",Resources);
  printf("BsaCompleteU:\t%08x\n",BsaCompleteU);
  printf("BsaCompleteL:\t%08x\n",BsaCompleteL);
  printf("BsaDef[0]:\t%08x/%08x\n",BsaDef[0].l,BsaDef[0].h);
  printf("CntPLL:\t%08x\n",CntPLL);
  printf("Cnt186M:\t%08x\n",Cnt186M);
  printf("CntIntvl:\t%08x\n",CntIntvl);
  printf("CntBRT:\t%08x\n",CntBRT);
}
