#include "Module134.hh"

#include "ModuleBase.hh"
#include "I2c134.hh"
#include "Mmcm.hh"
#include "Pgp3.hh"
#include "ChipAdcCore.hh"
#include "Jesd204b.hh"
#include "Fmc134Ctrl.hh"
//#include "FlashController.hh"
#include "OptFmc.hh"
#include "psdaq/aes-stream-drivers/DmaDriver.h"
#include "psdaq/aes-stream-drivers/AxiVersion.h"

#include "psdaq/mmhw/Pgp3Axil.hh"
#include "psdaq/mmhw/TriggerEventManager2.hh"
#include "psdaq/mmhw/TprCore.hh"
//#include "psdaq/mmhw/Xvc.hh"

//using Pds::Mmhw::AxiVersion;
using Pds::Mmhw::Pgp3Axil;
using Pds::Mmhw::RingBuffer;
using Pds::Mmhw::TriggerEventManager2;
using Pds::Mmhw::TriggerEventBuffer;
using Pds::Mmhw::TprCore;

#include <string>
#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <poll.h>

using std::string;

typedef volatile uint32_t vuint32_t;

namespace Pds {
  namespace HSD {

    class Module134::PrivateData {
    public:
      void dumpPgp         () const;
      //
      //  Low level API
      //
      //  Core registers
      ModuleBase  base                 ; // 0
      uint32_t    rsvd_800000[(0x800000-sizeof(base))>>2];
      //  App registers
      ChipAdcCore chip[2]              ; // 0x800000, 0x802000
      uint32_t    rsvd_810000[0xc000>>2];
      Fmc134Ctrl  fmc_ctrl             ; // 0x810000
      uint32_t    rsvd_820000[(0x10000-sizeof(fmc_ctrl))>>2];
      Mmcm        mmcm                 ; // 0x820000
      uint32_t    rsvd_840000[(0x20000-sizeof(mmcm))>>2];
      Mmhw::Reg   opt_fmc  [0x10000>>2]; // 0x840000
      Mmhw::Reg   qsfp0_i2c[0x10000>>2]; // 0x850000
      Mmhw::Reg   qsfp1_i2c[0x10000>>2]; // 0x860000
      Mmhw::Reg   surf_jesd0[0x10000>>2]; // 0x870000
      Mmhw::Reg   surf_jesd1[0x10000>>2]; // 0x880000
      TriggerEventManager2 tem          ; // 0x890000
      Mmhw::Reg   rsvd_900000[(0x70000-sizeof(tem))>>2];
      Mmhw::Reg   pgp_reg  [0x100000>>2]  ; // 0x900000
    };
  };
};

using namespace Pds::HSD;

Module134* Module134::create(int fd)
{
  // void* ptr = mmap(0, sizeof(Module134::PrivateData), PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  // if (ptr == MAP_FAILED) {
  //   perror("Failed to map");
  //   return 0;
  // }

  Module134* m = new Module134;
  m->p = 0;
  m->_fd = fd;

  Pds::Mmhw::Reg::set(fd);
  Pds::Mmhw::RegProxy::initialize(m->p, m->p->base.regProxy); // this hangs the CPU

  AxiVersion vsn;
  axiVersionGet(fd,&vsn);
  m->_vsn.FpgaVersion = vsn.firmwareVersion;
  memcpy((void*)(&m->_vsn.dnaValue),&vsn.dnaValue,16);
  memcpy((void*)(&m->_vsn.BuildStamp),&vsn.buildString,256);

  return m;
}

void Module134::setup_timing(bool lLoopback)
{
  p->base.tprLoopback = lLoopback ? 8:0;
  //
  //  Verify clock synthesizer is setup
  //  Necessary for timing and dual firefly channel pgp
  //
  {
    TprCore& tpr = p->base.tpr;
    double txclkr = tpr.txRefClockRate();
    printf("TxRefClk: %f MHz\n", txclkr);

    const_cast<Module134*>(this)->i2c().i2c_sw_control.dump();

    static const double TXCLKR_MIN = 185.;
    static const double TXCLKR_MAX = 187.;
    if (txclkr < TXCLKR_MIN ||
        txclkr > TXCLKR_MAX) {
      i2c_lock(I2cSwitch::LocalBus);  // ClkSynth is on local bus
      i2c().clksynth.setup(LCLSII);
      //      i2c().clksynth.setup(M64);
      i2c_unlock();

      usleep(100000);
      tpr.setLCLSII();
      tpr.resetRxPll();
      usleep(1000000);
      tpr.resetBB();

      ChipAdcReg& r0 = chip(0).reg;
      ChipAdcReg& r1 = chip(1).reg;
      r0.resetFb ();
      r0.resetDma();
      r1.resetDma();
      usleep(1000000);

      tpr.resetCounts();

      usleep(100000);

      tpr.resetRxPll();
      r0.resetFbPLL();

      optfmc().resetPgp();
    }
    else {
      tpr.resetCounts();
      usleep(100000);
      tpr.dump();
    }
    double rxclkr = tpr.rxRecClockRate();
    printf("RxRecClk: %f MHz\n", rxclkr);
  }
}

void Module134::_jesd_init(unsigned mode)
{
  Fmc134Ctrl& ctrl = jesdctl();
  Fmc134Cpld& cpld = i2c().fmc_cpld;
  Mmhw::Reg* jesd0  = &p->surf_jesd0[0];
  Mmhw::Reg* jesd1  = &p->surf_jesd1[0];
  while(1) {
      if (!ctrl.default_init(cpld, mode, true)) {
          usleep(2000000);
          unsigned dvalid=0;
          for(unsigned i=0; i<8; i++) {
              dvalid |= (jesd0[0x10+i]&2) ? (0x001<<i) : 0;
              dvalid |= (jesd1[0x10+i]&2) ? (0x100<<i) : 0;
          }
          if (dvalid == 0xffff)
              break;
          printf("dvalid: 0x%x\n",dvalid);
          break;
      }
      usleep(1000);
  }

  ctrl.dump();
}

void Module134::setup_jesd(bool lAbortOnErr,
                           std::string& adc0,
                           std::string& adc1,
                           bool lInternalTiming)
{
  i2c_lock(I2cSwitch::PrimaryFmc);
  Fmc134Cpld* cpld = &i2c().fmc_cpld;
  Fmc134Ctrl* ctrl = &p->fmc_ctrl;
  Mmhw::Reg* jesd0  = &p->surf_jesd0[0];
  Mmhw::Reg* jesd1  = &p->surf_jesd1[0];
  while (cpld->default_clocktree_init(lInternalTiming ?
                                      Fmc134Cpld::CLOCKTREE_CLKSRC_INTERNAL :
                                      Fmc134Cpld::CLOCKTREE_REFSRC_EXTERNAL)) {
    if (lAbortOnErr)
      abort();
    usleep(1000);
  }

  while (cpld->default_adc_init(Fmc134Cpld::FG_CAL,adc0,adc1)) {
    if (lAbortOnErr)
      abort();
    usleep(1000);
  }

  cpld->dump();
  jesd0[0] = 0xff;  // enable rx lanes
  jesd1[0] = 0xff;
  jesd0[4] = 0x27;  // subclass 1, replace enable, reset GTs, scramble enable
  jesd1[4] = 0x27;
  usleep(100);
  jesd0[4] = 0x23;  // remove reset GTs
  jesd1[4] = 0x23;

  _jesd_init(0);

  ctrl->dump();
  i2c_unlock();
}

Module134::~Module134()
{
}

#define PGPLANES 8
#define LPRINT(title,field) {                     \
      printf("\t%20.20s :",title);                \
      for(unsigned i=0; i<PGPLANES; i++)          \
          printf(" %11x",unsigned(pgp[i].field)); \
      printf("\n"); }

#define LPRBF(title,field,shift,mask) {                 \
      printf("\t%20.20s :",title);                      \
      for(unsigned i=0; i<PGPLANES; i++)                \
          printf(" %11x",(unsigned(pgp[i].field)>>shift)&mask); \
      printf("\n"); }

#define LPRVC(title,field) {                      \
      printf("\t%20.20s :",title);                \
      for(unsigned i=0; i<PGPLANES; i++)          \
        printf(" %2x %2x %2x %2x",                \
            unsigned(pgp[i].field##0),                    \
               unsigned(pgp[i].field##1),                 \
               unsigned(pgp[i].field##2),                 \
               unsigned(pgp[i].field##3) );               \
    printf("\n"); }

#define LPRFRQ(title,field) {                           \
      printf("\t%20.20s :",title);                      \
      for(unsigned i=0; i<PGPLANES; i++)                \
        printf(" %11.4f",double(pgp[i].field)*1.e-6);   \
      printf("\n"); }

void Module134::PrivateData::dumpPgp     () const
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
  LPRINT("rxInitCnt"      ,rxInitCnt);

  static const char* clock_name[] = {"PLL","RX","SYSREF","DEVCLK","GTREF","PGPREF","TIMREF",NULL};
  printf("%10.10s %10.10s SLOW FAST LOCK\n","Clock","Rate, MHz");
  for(unsigned i=0; i<7; i++)
    printf("%10.10s %10.5f    %c   %c   %c\n",
           clock_name[i], double(opt_fmc[2+i]&0x1fffffff)*1.e-6,
           opt_fmc[2+i]&(1<<29) ? 'Y':'.',
           opt_fmc[2+i]&(1<<30) ? 'Y':'.',
           opt_fmc[2+i]&(1<<31) ? 'Y':'.');
}

#undef LPRINT
#undef LPRBF
#undef LPRVC
#undef LPRFRQ

void Module134::dumpMap() const
{
  const char* cthis = reinterpret_cast<const char*>(p);
  I2c134& i2c = const_cast<Module134*>(this)->i2c();
#define OFFS(member) (reinterpret_cast<const char*>(&p->member)-cthis)
#define OFFP(pval  ) (reinterpret_cast<const char*>(&pval)-cthis)
  //  printf("AxiVersion     : 0x%lx\n", OFFS(base.version));
  //  printf("FlashController: 0x%lx\n", OFFS(base.flash));
  printf("I2cSwitch      : 0x%lx\n", OFFP(i2c.i2c_sw_control));
  printf("ClkSynth       : 0x%lx\n", OFFP(i2c.clksynth));
  printf("LocalCpld      : 0x%lx\n", OFFP(i2c.local_cpld));
  //  printf("FmcSpi         : 0x%x\n", &p->fmc_spi);
  //  printf("DmaCore        : 0x%lx\n", OFFS(base.dma_core));
  printf("TprCore        : 0x%lx\n", OFFS(base.tpr));
  printf("ChipAdcCore[0] : 0x%lx\n", OFFS(chip[0]));
  printf("ChipAdcCore[1] : 0x%lx\n", OFFS(chip[1]));
  printf("Fmc134Ctrl     : 0x%lx\n", OFFS(fmc_ctrl));
  printf("mmcm           : 0x%lx\n", OFFS(mmcm));
  printf("Pgp            : 0x%lx\n", OFFS(pgp_reg[0]));
  printf("Tem            : 0x%lx\n", OFFS(tem));
#undef OFFS
}

uint64_t Module134::device_dna() const
{
  uint64_t v = version().DeviceDnaHigh;
  v <<= 32;
  v |= version().DeviceDnaLow;
  return v;
}

void Module134::enable_test_pattern(TestPattern p)
{
  _jesd_init(unsigned(p));
}

void Module134::disable_test_pattern()
{
  _jesd_init(0);
}

// Update ID advertised on timing and pgp link

void Module134::set_local_id(unsigned bus)
{
  p->tem.xma().txId = ModuleBase::local_id(bus);
  optfmc    ().txId = bus;
  printf("***********************\n");
  printf("Programmed txId %08x\n", unsigned(optfmc().txId));
  printf("***********************\n");
}

unsigned Module134::remote_id() const { return p->tem.xma().rxId; }

void Module134::board_status()
{
    { const Pds::Mmhw::AxiVersion& v = version();
    printf("Axi Version [%p]: BuildStamp[%p]: %s\n",
           &v, &v.BuildStamp[0], v.buildStamp().c_str());
    printf("Dna: %08x%08x  Serial: %08x%08x\n",
           v.DeviceDnaHigh,
           v.DeviceDnaLow,
           v.FdSerialHigh,
           v.FdSerialLow ); }

  p->fmc_ctrl.dump();

  i2c_lock(I2cSwitch::LocalBus);
  { LocalCpld& v = i2c().local_cpld;
    printf("Local CPLD revision: 0x%x\n", v.revision());
    printf("Local CPLD GAaddr  : 0x%x\n", v.GAaddr  ());
    v.GAaddr(0); }

  printf("vtmon1 mfg:dev %x:%x\n", i2c().vtmon1.manufacturerId(), i2c().vtmon1.deviceId());
  printf("vtmon2 mfg:dev %x:%x\n", i2c().vtmon2.manufacturerId(), i2c().vtmon2.deviceId());
  printf("vtmon3 mfg:dev %x:%x\n", i2c().vtmon3.manufacturerId(), i2c().vtmon3.deviceId());

  i2c().vtmon1.dump();
  i2c().vtmon2.dump();
  i2c().vtmon3.dump();

  printf("imona/b\n");
  i2c().imona.dump();
  i2c().imonb.dump();
  i2c_unlock();

  i2c_lock(I2cSwitch::PrimaryFmc);
  { unsigned v;
    printf("FMC EEPROM:");
    for(unsigned i=0; i<32; i++) {
      v = i2c().eeprom[i];
      printf(" %02x", v&0xff);
    }
    printf("\n");
  }

  i2c().fmc_cpld.dump();

  i2c().fmc_cpld.enable_mon(true);
  printf("-- fmcadcmon --\n");
  FmcAdcMon(i2c().fmcadcmon.mon()).dump();

  printf("-- fmcvmon --\n");
  FmcVMon(i2c().fmcvmon.mon()).dump();
  i2c().fmc_cpld.enable_mon(false);

  i2c_unlock();
}

ChipAdcCore& Module134::chip   (unsigned ch) { return p->chip[ch]; }

void Module134::dumpRxAlign     () const { p->base.dumpRxAlign(); }
void Module134::dumpPgp         () const { p->dumpPgp(); }

void* Module134::reg() { return (void*)p; }

std::vector<Pgp*> Module134::pgp() {
  std::vector<Pgp*> v(0);
  while(1) {
    Pgp3Axil* pgp = reinterpret_cast<Pgp3Axil*>(p->pgp_reg);
    for(unsigned i=0; i<PGPLANES; i++)
      v.push_back(new Pgp3(pgp[i]));
    break;
  }
  return v;
}

TriggerEventManager2& Module134::tem() { return p->tem; }

Fmc134Ctrl& Module134::jesdctl() { return p->fmc_ctrl; }

OptFmc&     Module134::optfmc() { return *reinterpret_cast<OptFmc*>(p->opt_fmc); }

Mmcm&       Module134::mmcm() { return p->mmcm; }

TprCore&    Module134::tpr() { return p->base.tpr; }

Jesd204b& Module134::jesd(unsigned ch)
{ return *reinterpret_cast<Jesd204b*>(ch==0 ? p->surf_jesd0 : p->surf_jesd1); }

void   Module134::mon_start()
{
  i2c_lock(I2cSwitch::LocalBus);
  i2c().vtmon1.start();
  i2c().vtmon2.start();
  i2c().vtmon3.start();
  i2c().imona.start();
  i2c().imonb.start();
  i2c_unlock();
}

EnvMon Module134::mon() const
{
  i2c_lock(I2cSwitch::LocalBus);
  EnvMon v;
  Adt7411_Mon m;
  I2c134& i2c = const_cast<Module134*>(this)->i2c();
  m = i2c.vtmon1.mon();
  v.local12v = m.ain[3]*6.;
  v.edge12v  = m.ain[6]*6.;
  v.aux12v   = m.ain[7]*6.;
  m = i2c.vtmon2.mon();
  v.boardTemp = m.Tint;
  //  v.boardTemp = m.Text;
  v.local1_8v = m.ain[6];
  m = i2c.vtmon3.mon();
  v.fmc12v = m.ain[2]*6.;
  v.local2_5v = m.ain[6]*2.;
  v.local3_3v = m.ain[7]*2.;

  v.fmcPower   = i2c.imona.power_W();
  v.totalPower = i2c.imonb.power_W();
  i2c_unlock();

  return v;
}

I2c134& Module134::i2c()
{
  return *reinterpret_cast<I2c134*>(p->base.i2c_regs);
}

void Module134::i2c_lock  (I2cSwitch::Port port) const
{
  _sem_i2c.take();
  const_cast<Module134*>(this)->i2c().i2c_sw_control.select(port);
}
void Module134::i2c_unlock() const { _sem_i2c.give(); }

#if 0
Pds::Mmhw::Jtag& Module134::xvc()
{
  return p->base.xvc;
}
#endif
