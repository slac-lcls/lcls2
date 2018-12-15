#include "psdaq/hsd/Hsd3200.hh"

#include "psdaq/mmhw/AxiVersion.hh"
#include "psdaq/hsd/TprCore.hh"
#include "psdaq/hsd/RxDesc.hh"
#include "psdaq/hsd/ClkSynth.hh"
#include "psdaq/hsd/Mmcm.hh"
#include "psdaq/hsd/DmaCore.hh"
#include "psdaq/hsd/PhyCore.hh"
#include "psdaq/hsd/Pgp3.hh"
#include "psdaq/hsd/QABase.hh"
//#include "psdaq/mmhw/Pgp3Axil.hh"
#include "psdaq/mmhw/RingBuffer.hh"
#include "psdaq/mmhw/Xvc.hh"
#include "psdaq/hsd/I2cSwitch.hh"
#include "psdaq/hsd/Jesd204b.hh"
#include "psdaq/hsd/LocalCpld.hh"
#include "psdaq/hsd/Fmc134Cpld.hh"
#include "psdaq/hsd/Fmc134Ctrl.hh"
#include "psdaq/hsd/Adt7411.hh"
#include "psdaq/hsd/Ad7291.hh"
#include "psdaq/hsd/Tps2481.hh"
#include "psdaq/hsd/FexCfg.hh"
#include "psdaq/hsd/HdrFifo.hh"
#include "psdaq/hsd/PhaseMsmt.hh"
#include "psdaq/hsd/FlashController.hh"

using Pds::Mmhw::AxiVersion;
using Pds::Mmhw::RingBuffer;

#include <string>
#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <poll.h>

using std::string;

namespace Pds {
  namespace HSD {

    class Pgp3Axil {
    public:
      uint32_t  countReset;
      uint32_t  autoStatus;
      uint32_t  loopback;
      uint32_t  skpInterval;
      uint32_t  rxStatus; // phyRxActive, locLinkReady, remLinkReady
      uint32_t  cellErrCnt;
      uint32_t  linkDownCnt;
      uint32_t  linkErrCnt;
      uint32_t  remRxOflow; // +pause
      uint32_t  rxFrameCnt;
      uint32_t  rxFrameErrCnt;
      uint32_t  rxClkFreq;
      uint32_t  rxOpCodeCnt;
      uint32_t  rxOpCodeLast;
      uint32_t  rxOpCodeNum;
      uint32_t  rsvd_3C;
      uint32_t  rsvd_40[0x10];
      // tx
      uint32_t  cntrl; // flowCntDis, txDisable
      uint32_t  txStatus; // phyTxActive, linkReady
      uint32_t  rsvd_88;
      uint32_t  locStatus; // locOflow, locPause
      uint32_t  txFrameCnt;
      uint32_t  txFrameErrCnt;
      uint32_t  rsvd_98;
      uint32_t  txClkFreq;
      uint32_t  txOpCodeCnt;
      uint32_t  txOpCodeLast;
      uint32_t  txOpCodeNum;
      uint32_t  rsvd_AC;
      uint32_t  rsvd_B0[0x14];
    };

    class Hsd3200::PrivateData {
    public:
      //  Initialize busses
      void init();

      //  Initialize clock tree and IO training
      void fmc_init();

      void enable_test_pattern(TestPattern);
      void disable_test_pattern();
      void enable_cal ();
      void disable_cal();
      void setAdcMux(bool     interleave,
                     unsigned channels);

      void setRxAlignTarget(unsigned);
      void setRxResetLength(unsigned);
      void dumpRxAlign     () const;
      void dumpPgp         () const;
      //
      //  Low level API
      //
    public:
      AxiVersion version;
      uint32_t rsvd_to_0x08000[(0x8000-sizeof(AxiVersion))/4];

      FlashController      flash;
      uint32_t rsvd_to_0x10000[(0x8000-sizeof(FlashController))/4];

      I2cSwitch  i2c_sw_control;  // 0x10000
      ClkSynth   clksynth;        // 0x10400
      LocalCpld  local_cpld;      // 0x10800
      Adt7411    vtmon1;          // 0x10C00
      Adt7411    vtmon2;          // 0x11000
      Adt7411    vtmon3;          // 0x11400
      Tps2481    imona;           // 0x11800
      Tps2481    imonb;           // 0x11C00
      Ad7291     fmcadcmon;       // 0x12000
      Ad7291     fmcvmon;         // 0x12400
      Fmc134Cpld fmc_cpld;        // 0x12800 
      uint32_t   eeprom[0x100];   // 0x12C00
      uint32_t   rsvd_to_0x20000[(0x10000-12*0x400)/4];

      // DMA
      DmaCore           dma_core; // 0x20000
      uint32_t rsvd_to_0x30000[(0x10000-sizeof(DmaCore))/4];

      // PHY
      PhyCore           phy_core; // 0x30000
      uint32_t rsvd_to_0x31000[(0x1000-sizeof(PhyCore))/4];

      // GTH
      uint32_t gthAlign[10];     // 0x31000
      uint32_t rsvd_to_0x31100  [54];
      uint32_t gthAlignTarget;
      uint32_t gthAlignLast;
      uint32_t rsvd_to_0x31800[(0x800-0x108)/4];
      uint32_t gthDrp[0x200];

      // XVC
      Pds::Mmhw::Jtag    xvc;
      uint32_t rsvd_to_0x40000[(0xE000-sizeof(xvc))/4];

      // Timing
      Pds::HSD::TprCore  tpr;     // 0x40000
      uint32_t rsvd_to_0x50000  [(0x10000-sizeof(Pds::HSD::TprCore))/4];

      RingBuffer         ring0;   // 0x50000
      uint32_t rsvd_to_0x60000  [(0x10000-sizeof(RingBuffer))/4];

      RingBuffer         ring1;   // 0x60000
      uint32_t rsvd_to_0x70000  [(0x10000-sizeof(RingBuffer))/4];
      uint32_t rsvd_to_0x80000  [0x10000/4];

      //  App registers
      QABase     base;             // 0x80000
      uint32_t   rsvd_to_0x80800  [(0x800-sizeof(base))/4];

      Mmcm       mmcm;             // 0x80800
      Fmc134Ctrl fmc_ctrl;         // 0x81000
      uint32_t   rsvd_to_0x82800 [(0x1800-sizeof(fmc_ctrl))/4];
      HdrFifo    hdr_fifo[4];      // 0x82800
      PhaseMsmt  trg_phase[2];
      uint32_t   rsvd_to_0x88000  [(0x5800-4*sizeof(HdrFifo)-2*sizeof(PhaseMsmt))/4];

      FexCfg     fex_chan[4];      // 0x88000
      uint32_t   rsvd_to_0x90000  [(0x8000-4*sizeof(FexCfg))/4];

      uint32_t    pgp_reg[0x4000>>2];

      uint32_t    opt_fmc  [0x1000>>2];
      uint32_t    qsfp0_i2c[0x400>>2];
      uint32_t    qsfp1_i2c[0x400>>2];
    };
  };
};

using namespace Pds::HSD;

Hsd3200* Hsd3200::create(int fd)
{
  void* ptr = mmap(0, sizeof(Pds::HSD::Hsd3200::PrivateData), PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    perror("Failed to map");
    return 0;
  }

  Pds::HSD::Hsd3200* m = new Pds::HSD::Hsd3200;
  m->p = reinterpret_cast<Pds::HSD::Hsd3200::PrivateData*>(ptr);
  m->_fd = fd;

  return m;
}

void Hsd3200::setup_timing(TimingType timing)
{
  Hsd3200* m = this;

  //
  //  Verify clock synthesizer is setup
  //
  if (timing != EXTERNAL) {
    timespec tvb;
    clock_gettime(CLOCK_REALTIME,&tvb);
    unsigned vvb = m->tpr().TxRefClks;

    usleep(10000);

    timespec tve;
    clock_gettime(CLOCK_REALTIME,&tve);
    unsigned vve = m->tpr().TxRefClks;
    
    double dt = double(tve.tv_sec-tvb.tv_sec)+1.e-9*(double(tve.tv_nsec)-double(tvb.tv_nsec));
    double txclkr = 16.e-6*double(vve-vvb)/dt;
    printf("TxRefClk: %f MHz\n", txclkr);

    static const double TXCLKR_MIN = 185.;
    static const double TXCLKR_MAX = 187.;
    if (txclkr < TXCLKR_MIN ||
        txclkr > TXCLKR_MAX) {
      m->fmc_clksynth_setup(timing);

      usleep(100000);
      m->tpr().setLCLSII();
      m->tpr().resetRxPll();
      usleep(1000000);
      m->tpr().resetBB();

      /*
      m->p->fmc_ctrl.resetFbPLL();
      usleep(1000000);
      m->p->fmc_ctrl.resetFb ();
      m->p->fmc_ctrl.resetDma();
      */
      usleep(1000000);

      m->tpr().resetCounts();

      usleep(100000);
      m->fmc_init();
    }
  }
}

Hsd3200::~Hsd3200()
{
}

int Hsd3200::read(uint32_t* data, unsigned data_size)
{
  RxDesc* desc = new RxDesc(data,data_size);
  int nw = ::read(_fd, desc, sizeof(*desc));
  delete desc;

  nw *= sizeof(uint32_t);
  if (nw>=32)
    data[7] = nw - 8*sizeof(uint32_t);

  return nw;
}

void Hsd3200::PrivateData::init()
{
  // setup I2C switches
}

void Hsd3200::PrivateData::fmc_init()
{
  fmc_ctrl.adc_pins = 1; // Assert SYNC

  // setup PLL, Clock distribution
  fmc_cpld.initialize(true,true);
  fmc_cpld.dump      ();

  // enable scrambling in JESD204B core
  fmc_ctrl.scramble = 1;
  fmc_cpld.enable_adc_prbs(true);

  fmc_ctrl.xcvr = 0;// Stop hold TAP values, CDR, and AGCs
  fmc_ctrl.xcvr = 1;// Assert Transceiver Reset
  fmc_ctrl.xcvr = 0;// Release Tranceiver Reset
  usleep(100000);   // Wait for MGTs to adapt

  fmc_ctrl.xcvr = 0x1ff00;// Hold TAP values, CDR, and AGCs
  fmc_cpld.enable_adc_prbs(false);
  usleep(100000);   // Wait for QPLLs to lock

  fmc_ctrl.xcvr = 0x1ff02;// Assert DIV2 reset
  fmc_ctrl.xcvr = 0x1ff00;// Release DIV2 reset

  unsigned v = fmc_ctrl.status;
  printf("QPLLS %s locked [%x]\n", (v & 0xF00000) == 0xF00000 ? "are" : "NOT", v);

  fmc_ctrl.xcvr = 0x1ff10;// Enable transceiver alignment

  v = fmc_ctrl.status;
  if (((v>>16)&3)==3)
    printf("ADC0 Aligned\n");
  else
    printf("ADC0 Failed Bit Alignment\n");
  
  if (((v>>18)&3)==3)
    printf("ADC1 Aligned\n");
  else
    printf("ADC1 Failed Bit Alignment\n");

  v = fmc_ctrl.adc_val;
  if (v == 0xf)
    printf("Initial Lane Alignment Complete\n");
  else
    printf("Initial Lane Alignment FAILED [0x%x]\n", v);
}

void Hsd3200::PrivateData::enable_test_pattern(TestPattern p)
{
}

void Hsd3200::PrivateData::disable_test_pattern()
{
}

void Hsd3200::PrivateData::enable_cal()
{
}

void Hsd3200::PrivateData::disable_cal()
{
}

void Hsd3200::PrivateData::setRxAlignTarget(unsigned t)
{
  unsigned v = gthAlignTarget;
  v &= ~0x3f;
  v |= (t&0x3f);
  gthAlignTarget = v;
}

void Hsd3200::PrivateData::setRxResetLength(unsigned len)
{
  unsigned v = gthAlignTarget;
  v &= ~0xf0000;
  v |= (len&0xf)<<16;
  gthAlignTarget = v;
}
 
void Hsd3200::PrivateData::dumpRxAlign     () const
{
  printf("\nTarget: %u\tRstLen: %u\tLast: %u\n",
         gthAlignTarget&0x7f,
         (gthAlignTarget>>16)&0xf, 
         gthAlignLast&0x7f);
  for(unsigned i=0; i<128; i++) {
    printf(" %04x",(gthAlign[i/2] >> (16*(i&1)))&0xffff);
    if ((i%10)==9) printf("\n");
  }
  printf("\n");
}

#define LPRINT(title,field) {                     \
      printf("\t%20.20s :",title);                \
      for(unsigned i=0; i<4; i++)                 \
        printf(" %11x",pgp[i].field);             \
      printf("\n"); }
    
#define LPRBF(title,field,shift,mask) {                 \
      printf("\t%20.20s :",title);                      \
      for(unsigned i=0; i<4; i++)                       \
        printf(" %11x",(pgp[i].field>>shift)&mask);     \
      printf("\n"); }
    
#define LPRVC(title,field) {                      \
      printf("\t%20.20s :",title);                \
      for(unsigned i=0; i<4; i++)                 \
        printf(" %2x %2x %2x %2x",                \
               pgp[i].field##0,                   \
             pgp[i].field##1,                     \
             pgp[i].field##2,                     \
             pgp[i].field##3 );                   \
    printf("\n"); }

#define LPRFRQ(title,field) {                           \
      printf("\t%20.20s :",title);                      \
      for(unsigned i=0; i<4; i++)                       \
        printf(" %11.4f",double(pgp[i].field)*1.e-6);   \
      printf("\n"); }
    
void Hsd3200::PrivateData::dumpPgp     () const
{
  const Pgp3Axil* pgp = reinterpret_cast<const Pgp3Axil*>(pgp_reg);
  LPRINT("loopback"       ,loopback);
  LPRINT("skpInterval"    ,skpInterval);
  LPRBF ("localLinkReady" ,rxStatus,1,1);
  LPRBF ("remoteLinkReady",rxStatus,2,1);
  LPRINT("framesRxErr"    ,rxFrameErrCnt);
  LPRINT("framesRx"       ,rxFrameCnt);
  LPRINT("framesTxErr"    ,txFrameErrCnt);
  LPRINT("framesTx"       ,txFrameCnt);
  LPRFRQ("rxClkFreq"      ,rxClkFreq);
  LPRFRQ("txClkFreq"      ,txClkFreq);
  LPRINT("lastTxOp"       ,txOpCodeLast);
  LPRINT("lastRxOp"       ,rxOpCodeLast);
  LPRINT("nTxOps"         ,txOpCodeCnt);
  LPRINT("nRxOps"         ,rxOpCodeCnt);

  printf("optFmc: %08x %08x\n", opt_fmc[0], opt_fmc[1]);
}

#undef LPRINT
#undef LPRBF
#undef LPRVC
#undef LPRFRQ

void Hsd3200::dumpBase() const
{
}

void Hsd3200::dumpMap() const
{
  const char* cthis = reinterpret_cast<const char*>(p);
#define OFFS(member) (reinterpret_cast<const char*>(&p->member)-cthis)
  printf("AxiVersion     : 0x%lx\n", OFFS(version));
  printf("FlashController: 0x%lx\n", OFFS(flash));
  printf("I2cSwitch      : 0x%lx\n", OFFS(i2c_sw_control));
  printf("ClkSynth       : 0x%lx\n", OFFS(clksynth));
  printf("LocalCpld      : 0x%lx\n", OFFS(local_cpld));
  //  printf("FmcSpi         : 0x%x\n", &p->fmc_spi);
  printf("DmaCore        : 0x%lx\n", OFFS(dma_core));
  printf("TprCore        : 0x%lx\n", OFFS(tpr));
  printf("Fmc134Ctrl     : 0x%lx\n", OFFS(fmc_ctrl));
  printf("HdrFifo        : 0x%lx\n", OFFS(hdr_fifo[0]));
  printf("FexCfg         : 0x%lx\n", OFFS(fex_chan[0]));
  printf("Pgp            : 0x%lx\n", OFFS(pgp_reg[0]));
#undef OFFS
}

void Hsd3200::PrivateData::setAdcMux(bool     interleave,
                                    unsigned channels)
{
}

void Hsd3200::init() { p->init(); }

void Hsd3200::fmc_init(TimingType) 
{
  p->i2c_sw_control.select(I2cSwitch::PrimaryFmc);
  p->fmc_init(); 
}

void Hsd3200::fmc_dump() {
}

void Hsd3200::fmc_clksynth_setup(TimingType timing)
{
  p->i2c_sw_control.select(I2cSwitch::LocalBus);  // ClkSynth is on local bus
  p->clksynth.setup(timing);
  p->clksynth.dump ();
}

uint64_t Hsd3200::device_dna() const
{
  uint64_t v = p->version.DeviceDnaHigh;
  v <<= 32;
  v |= p->version.DeviceDnaLow;
  return v;
}

void Hsd3200::board_status()
{
  printf("Axi Version [%p]: BuildStamp[%p]: %s\n", 
         &(p->version), &(p->version.BuildStamp[0]), p->version.buildStamp().c_str());

  printf("Dna: %08x%08x  Serial: %08x%08x\n",
         p->version.DeviceDnaHigh,
         p->version.DeviceDnaLow,
         p->version.FdSerialHigh,
         p->version.FdSerialLow );

  printf("optFmc: %08x %08x\n", p->opt_fmc[0], p->opt_fmc[1]);

  p->fmc_ctrl.dump();

  p->i2c_sw_control.select(I2cSwitch::LocalBus);
  p->i2c_sw_control.dump();
  
  printf("Local CPLD revision: 0x%x\n", p->local_cpld.revision());
  printf("Local CPLD GAaddr  : 0x%x\n", p->local_cpld.GAaddr  ());
  p->local_cpld.GAaddr(0);

  printf("vtmon1 mfg:dev %x:%x\n", p->vtmon1.manufacturerId(), p->vtmon1.deviceId());
  printf("vtmon2 mfg:dev %x:%x\n", p->vtmon2.manufacturerId(), p->vtmon2.deviceId());
  printf("vtmon3 mfg:dev %x:%x\n", p->vtmon3.manufacturerId(), p->vtmon3.deviceId());

  p->vtmon1.dump();
  p->vtmon2.dump();
  p->vtmon3.dump();

  printf("imona/b\n");
  p->imona.dump();
  p->imonb.dump();

  p->i2c_sw_control.select(I2cSwitch::PrimaryFmc);
  p->i2c_sw_control.dump();

  { unsigned v;
    printf("FMC EEPROM:");
    for(unsigned i=0; i<32; i++) {
      v = p->eeprom[i];
      printf(" %02x", v&0xff);
    }
    printf("\n");
  }

  p->fmc_cpld.dump();

  p->fmc_cpld.enable_mon(true);
  printf("-- fmcadcmon --\n");
  FmcAdcMon(p->fmcadcmon.mon()).dump();

  printf("-- fmcvmon --\n");
  FmcVMon(p->fmcvmon.mon()).dump();
  p->fmc_cpld.enable_mon(false);
}

void Hsd3200::enable_test_pattern(TestPattern t) { p->enable_test_pattern(t); }

void Hsd3200::disable_test_pattern() { p->disable_test_pattern(); }

void Hsd3200::clear_test_pattern_errors() {
  for(unsigned i=0; i<4; i++) {
    p->fex_chan[i]._test_pattern_errors  = 0;
    p->fex_chan[i]._test_pattern_errbits = 0;
  }
}

void Hsd3200::enable_cal () { p->enable_cal(); }

void Hsd3200::disable_cal() { p->disable_cal(); }

void Hsd3200::setAdcMux(unsigned channels)
{
}

void Hsd3200::setAdcMux(bool     interleave,
                       unsigned channels) 
{}

const Pds::Mmhw::AxiVersion& Hsd3200::version() const { return p->version; }
Pds::HSD::TprCore&    Hsd3200::tpr    () { return p->tpr; }

void Hsd3200::setRxAlignTarget(unsigned v) { p->setRxAlignTarget(v); }
void Hsd3200::setRxResetLength(unsigned v) { p->setRxResetLength(v); }
void Hsd3200::dumpRxAlign     () const { p->dumpRxAlign(); }
void Hsd3200::dumpPgp         () const { p->dumpPgp(); }

void Hsd3200::sample_init(unsigned length, 
                         unsigned delay,
                         unsigned prescale)
{
  p->base.init();
  p->base.samples  = length;
  p->base.prescale = (delay<<6) | (prescale&0x3f);

  p->dma_core.init(32+48*length);

  p->dma_core.dump();

  //  p->dma.setEmptyThr(emptyThr);
  //  p->base.dmaFullThr=fullThr;

  //  flush out all the old
  { printf("flushing\n");
    unsigned nflush=0;
    uint32_t* data = new uint32_t[1<<20];
    RxDesc* desc = new RxDesc(data,1<<20);
    pollfd pfd;
    pfd.fd = _fd;
    pfd.events = POLLIN;
    while(poll(&pfd,1,0)>0) { 
      ::read(_fd, desc, sizeof(*desc));
      nflush++;
    }
    delete[] data;
    delete desc;
    printf("done flushing [%u]\n",nflush);
  }
    
  p->base.resetCounts();
}

void Hsd3200::trig_daq   (unsigned partition)
{
  p->base.setupDaq(partition);
}

void Hsd3200::trig_shift (unsigned shift)
{
  p->base.setTrigShift(shift);
}

void Hsd3200::start()
{
  p->base.start();
}

void Hsd3200::stop()
{
  p->base.stop();
  p->dma_core.dump();
}

unsigned Hsd3200::get_offset(unsigned channel)
{ return 0;
}

unsigned Hsd3200::get_gain(unsigned channel)
{ return 0;
}

void Hsd3200::set_offset(unsigned channel, unsigned value)
{
}

void Hsd3200::set_gain(unsigned channel, unsigned value)
{
}

void Hsd3200::clocktree_sync()
{
}

void Hsd3200::sync()
{
}

void* Hsd3200::reg() { return (void*)p; }

std::vector<Pgp*> Hsd3200::pgp() {
  std::vector<Pgp*> v(0);
  while(1) {
    // Pgp3Axil* pgp = reinterpret_cast<Pgp3Axil*>(p->pgp_reg);
    // for(unsigned i=0; i<4; i++)
    //   v.push_back(new Pgp3(pgp[i]));
    break;
  }
  return v;
}

Pds::Mmhw::Jtag* Hsd3200::xvc() { return &p->xvc; }

FexCfg* Hsd3200::fex() { return &p->fex_chan[0]; }

HdrFifo* Hsd3200::hdrFifo() { return &p->hdr_fifo[0]; }

uint32_t* Hsd3200::trgPhase() { return NULL; }

void   Hsd3200::mon_start()
{
  p->i2c_sw_control.select(I2cSwitch::LocalBus);
  p->vtmon1.start();
  p->vtmon2.start();
  p->vtmon3.start();
  p->imona.start();
  p->imonb.start();
}

EnvMon Hsd3200::mon() const
{
  p->i2c_sw_control.select(I2cSwitch::LocalBus);
  EnvMon v;
  Adt7411_Mon m;
  m = p->vtmon1.mon();
  v.local12v = m.ain[3]*6.;
  v.edge12v  = m.ain[6]*6.;
  v.aux12v   = m.ain[7]*6.;
  m = p->vtmon2.mon();
  v.boardTemp = m.Tint;
  v.local1_8v = m.ain[6];
  m = p->vtmon3.mon();
  v.fmc12v = m.ain[2]*6.;
  v.local2_5v = m.ain[6]*2.;
  v.local3_3v = m.ain[7]*2.;

  v.fmcPower   = p->imona.power_W();
  v.totalPower = p->imonb.power_W();
  return v;
}

