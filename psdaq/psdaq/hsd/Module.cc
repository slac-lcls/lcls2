#include "psdaq/hsd/Module.hh"

#include "psdaq/mmhw/AxiVersion.hh"
#include "psdaq/mmhw/RegProxy.hh"
#include "psdaq/hsd/TprCore.hh"
#include "psdaq/hsd/RxDesc.hh"
#include "psdaq/hsd/ClkSynth.hh"
#include "psdaq/hsd/Mmcm.hh"
#include "psdaq/hsd/DmaCore.hh"
#include "psdaq/hsd/PhyCore.hh"
#include "psdaq/hsd/Pgp2b.hh"
#include "psdaq/mmhw/Pgp2bAxi.hh"
#include "psdaq/hsd/Pgp3.hh"
#include "psdaq/mmhw/Pgp3Axil.hh"
#include "psdaq/mmhw/RingBuffer.hh"
#include "psdaq/mmhw/Xvc.hh"
#include "psdaq/hsd/I2cSwitch.hh"
#include "psdaq/hsd/LocalCpld.hh"
#include "psdaq/hsd/FmcSpi.hh"
#include "psdaq/hsd/QABase.hh"
#include "psdaq/hsd/Adt7411.hh"
#include "psdaq/hsd/Tps2481.hh"
#include "psdaq/hsd/AdcCore.hh"
#include "psdaq/hsd/AdcSync.hh"
#include "psdaq/hsd/FmcCore.hh"
#include "psdaq/hsd/FmcCoreVoid.hh"
#include "psdaq/hsd/FexCfg.hh"
#include "psdaq/hsd/HdrFifo.hh"
#include "psdaq/hsd/PhaseMsmt.hh"
#include "psdaq/hsd/FlashController.hh"

using Pds::Mmhw::AxiVersion;
using Pds::Mmhw::Pgp2bAxi;
using Pds::Mmhw::Pgp3Axil;
using Pds::Mmhw::RingBuffer;

#include <string>
#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <sys/mman.h>
#include <poll.h>
#include <string.h>

using std::string;

#define GUARD(f) { _sem_i2c.take(); f; _sem_i2c.give(); }

namespace Pds {
  namespace HSD {
    class Module::PrivateData {
    public:
      //  Initialize busses
      void init();

      //  Initialize clock tree and IO training
      void fmc_init(TimingType);

      int  train_io(unsigned);

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

      I2cSwitch i2c_sw_control;  // 0x10000
      ClkSynth  clksynth;        // 0x10400
      LocalCpld local_cpld;      // 0x10800
      Adt7411   vtmon1;          // 0x10C00
      Adt7411   vtmon2;          // 0x11000
      Adt7411   vtmon3;          // 0x11400
      Tps2481   imona;           // 0x11800
      Tps2481   imonb;           // 0x11C00
      Adt7411   vtmona;          // 0x12000
      FmcSpi    fmc_spi;         // 0x12400
      uint32_t  eeprom[0x100];   // 0x12800
      uint32_t  rsvd_to_0x18000[(0x08000-13*0x400)/4];
      uint32_t  regProxy[(0x08000)/4];

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
      uint32_t rsvd_to_0x32000[(0x1000-0x108)/4];

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
      QABase   base;             // 0x80000
      uint32_t rsvd_to_0x80800  [(0x800-sizeof(QABase))/4];

      Mmcm     mmcm;             // 0x80800
      FmcCore  fmca_core;        // 0x81000
      AdcCore  adca_core;        // 0x81400
      //      FmcCore  fmcb_core;        // 0x81800
      FmcCoreVoid  fmcb_core;        // 0x81800
      AdcCore  adcb_core;        // 0x81C00
      AdcSync  adc_sync;         // 0x82000
      HdrFifo  hdr_fifo[4];      // 0x82800
      PhaseMsmt trg_phase[2];
      uint32_t rsvd_to_0x88000  [(0x5800-4*sizeof(HdrFifo)-2*sizeof(PhaseMsmt))/4];

      FexCfg   fex_chan[4];      // 0x88000
      uint32_t rsvd_to_0x90000  [(0x8000-4*sizeof(FexCfg))/4];

      //  Pgp (optional)  
      //      Pgp2bAxi pgp[4];           // 0x90000
      //      uint32_t pgp_fmc1;
      //      uint32_t pgp_fmc2;
      uint32_t pgp_reg[0x4000>>2];
      uint32_t auxStatus;
      uint32_t auxControl;
    };
  };
};

using namespace Pds::HSD;

Module* Module::create(int fd)
{
  void* ptr = mmap(0, sizeof(Pds::HSD::Module::PrivateData), PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    perror("Failed to map");
    return 0;
  }

  Pds::HSD::Module* m = new Pds::HSD::Module;
  m->p = reinterpret_cast<Pds::HSD::Module::PrivateData*>(ptr);
  m->_fd = fd;

  Pds::Mmhw::RegProxy::initialize(m->p, m->p->regProxy);

  return m;
}

Module* Module::create(int fd, TimingType timing)
{
  Module* m = create(fd);

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

    static const double TXCLKR_MIN[] = { 118., 185., -1.00, 185., 185., 185. };
    static const double TXCLKR_MAX[] = { 120., 187., 1000., 187., 187., 187. };
    if (txclkr < TXCLKR_MIN[timing] ||
        txclkr > TXCLKR_MAX[timing]) {

      m->fmc_clksynth_setup(timing);

      usleep(100000);
      if (timing==LCLS)
        m->tpr().setLCLS();
      else
        m->tpr().setLCLSII();
      m->tpr().resetRxPll();
      usleep(1000000);
      m->tpr().resetBB();

      m->p->base.resetFbPLL();
      usleep(1000000);
      m->p->base.resetFb ();
      m->p->base.resetDma();
      usleep(1000000);

      m->tpr().resetCounts();

      usleep(100000);
      m->fmc_init(timing);
      m->train_io(0);
    }
  }

  return m;
}

Module::~Module()
{
}

int Module::read(uint32_t* data, unsigned data_size)
{
  RxDesc* desc = new RxDesc(data,data_size);
  int nw = ::read(_fd, desc, sizeof(*desc));
  delete desc;

  nw *= sizeof(uint32_t);
  if (nw>=32)
    data[7] = nw - 8*sizeof(uint32_t);

  return nw;
}

enum 
{
        FMC12X_INTERNAL_CLK = 0,                                                /*!< FMC12x_init() configure the FMC12x for internal clock operations */
        FMC12X_EXTERNAL_CLK = 1,                                                /*!< FMC12x_init() configure the FMC12x for external clock operations */
        FMC12X_EXTERNAL_REF = 2,                                                /*!< FMC12x_init() configure the FMC12x for external reference operations */

        FMC12X_VCXO_TYPE_2500MHZ = 0,                                   /*!< FMC12x_init() Vco on the card is 2.5GHz */
        FMC12X_VCXO_TYPE_2200MHZ = 1,                                   /*!< FMC12x_init() Vco on the card is 2.2GHz */
        FMC12X_VCXO_TYPE_2000MHZ = 2,                                   /*!< FMC12x_init() Vco on the card is 2.0GHz */
};

enum 
{
        CLOCKTREE_CLKSRC_EXTERNAL = 0,                                                  /*!< FMC12x_clocktree_init() configure the clock tree for external clock operations */
        CLOCKTREE_CLKSRC_INTERNAL = 1,                                                  /*!< FMC12x_clocktree_init() configure the clock tree for internal clock operations */
        CLOCKTREE_CLKSRC_EXTREF   = 2,                                                  /*!< FMC12x_clocktree_init() configure the clock tree for external reference operations */

        CLOCKTREE_VCXO_TYPE_2500MHZ = 0,                                                /*!< FMC12x_clocktree_init() Vco on the card is 2.5GHz */
        CLOCKTREE_VCXO_TYPE_2200MHZ = 1,                                                /*!< FMC12x_clocktree_init() Vco on the card is 2.2GHz */
        CLOCKTREE_VCXO_TYPE_2000MHZ = 2,                                                /*!< FMC12x_clocktree_init() Vco on the card is 2.0GHz */
        CLOCKTREE_VCXO_TYPE_1600MHZ = 3,                                                /*!< FMC12x_clocktree_init() Vco on the card is 1.6 GHZ */
        CLOCKTREE_VCXO_TYPE_BUILDIN = 4,                                                /*!< FMC12x_clocktree_init() Vco on the card is the AD9517 build in VCO */
        CLOCKTREE_VCXO_TYPE_2400MHZ = 5                                                 /*!< FMC12x_clocktree_init() Vco on the card is 2.4GHz */
};

enum 
{       // clock sources
        CLKSRC_EXTERNAL_CLK = 0,                                                /*!< FMC12x_cpld_init() external clock. */
        CLKSRC_INTERNAL_CLK_EXTERNAL_REF = 3,                   /*!< FMC12x_cpld_init() internal clock / external reference. */
        CLKSRC_INTERNAL_CLK_INTERNAL_REF = 6,                   /*!< FMC12x_cpld_init() internal clock / internal reference. */

        // sync sources
        SYNCSRC_EXTERNAL_TRIGGER = 0,                                   /*!< FMC12x_cpld_init() external trigger. */
        SYNCSRC_HOST = 1,                                                               /*!< FMC12x_cpld_init() software trigger. */
        SYNCSRC_CLOCK_TREE = 2,                                                 /*!< FMC12x_cpld_init() signal from the clock tree. */
        SYNCSRC_NO_SYNC = 3,                                                    /*!< FMC12x_cpld_init() no synchronization. */

        // FAN enable bits
        FAN0_ENABLED = (0<<4),                                                  /*!< FMC12x_cpld_init() FAN 0 is enabled */
        FAN1_ENABLED = (0<<5),                                                  /*!< FMC12x_cpld_init() FAN 1 is enabled */
        FAN2_ENABLED = (0<<6),                                                  /*!< FMC12x_cpld_init() FAN 2 is enabled */
        FAN3_ENABLED = (0<<7),                                                  /*!< FMC12x_cpld_init() FAN 3 is enabled */
        FAN0_DISABLED = (1<<4),                                                 /*!< FMC12x_cpld_init() FAN 0 is disabled */
        FAN1_DISABLED = (1<<5),                                                 /*!< FMC12x_cpld_init() FAN 1 is disabled */
        FAN2_DISABLED = (1<<6),                                                 /*!< FMC12x_cpld_init() FAN 2 is disabled */
        FAN3_DISABLED = (1<<7),                                                 /*!< FMC12x_cpld_init() FAN 3 is disabled */

        // LVTTL bus direction (HDMI connector)
        DIR0_INPUT      = (0<<0),                                                       /*!< FMC12x_cpld_init() DIR 0 is input */
        DIR1_INPUT      = (0<<1),                                                       /*!< FMC12x_cpld_init() DIR 1 is input */
        DIR2_INPUT      = (0<<2),                                                       /*!< FMC12x_cpld_init() DIR 2 is input */
        DIR3_INPUT      = (0<<3),                                                       /*!< FMC12x_cpld_init() DIR 3 is input */
        DIR0_OUTPUT     = (1<<0),                                                       /*!< FMC12x_cpld_init() DIR 0 is output */
        DIR1_OUTPUT     = (1<<1),                                                       /*!< FMC12x_cpld_init() DIR 1 is output */
        DIR2_OUTPUT     = (1<<2),                                                       /*!< FMC12x_cpld_init() DIR 2 is output */
        DIR3_OUTPUT     = (1<<3),                                                       /*!< FMC12x_cpld_init() DIR 3 is output */
};

void Module::PrivateData::init()
{
  i2c_sw_control.select(I2cSwitch::PrimaryFmc); 
  fmc_spi.initSPI();
  i2c_sw_control.select(I2cSwitch::SecondaryFmc); 
  fmc_spi.initSPI();
}

void Module::PrivateData::fmc_init(TimingType timing)
{
// if(FMC12x_init(AddrSipFMC12xBridge, AddrSipFMC12xClkSpi, AddrSipFMC12xAdcSpi, AddrSipFMC12xCpldSpi, AddrSipFMC12xAdcPhy, 
//                modeClock, cardType, GA, typeVco, carrierKC705)!=FMC12X_ERR_OK) {

#if 1
  const uint32_t clockmode = FMC12X_EXTERNAL_REF;
#else
  const uint32_t clockmode = FMC12X_INTERNAL_CLK;
#endif

  //  uint32_t clksrc_cpld;
  uint32_t clksrc_clktree;
  uint32_t vcotype = 0; // default 2500 MHz

  if(clockmode==FMC12X_INTERNAL_CLK) {
    //    clksrc_cpld    = CLKSRC_INTERNAL_CLK_INTERNAL_REF;
    clksrc_clktree = CLOCKTREE_CLKSRC_INTERNAL;
  }
  else if(clockmode==FMC12X_EXTERNAL_REF) {
    //    clksrc_cpld    = CLKSRC_INTERNAL_CLK_EXTERNAL_REF;
    clksrc_clktree = CLOCKTREE_CLKSRC_EXTREF;
  }
  else {
    //    clksrc_cpld    = CLKSRC_EXTERNAL_CLK;
    clksrc_clktree = CLOCKTREE_CLKSRC_EXTERNAL;
  }

  if (!fmca_core.present()) {
    printf("FMC card A not present\n");
    printf("FMC init failed!\n");
    return;
  }

  {
    printf("FMC card A initializing\n");
    i2c_sw_control.select(I2cSwitch::PrimaryFmc); 
    if (fmc_spi.cpld_init())
      printf("cpld_init failed!\n");
    if (fmc_spi.clocktree_init(clksrc_clktree, vcotype, timing))
      printf("clocktree_init failed!\n");
  }

  if (fmcb_core.present()) {
    printf("FMC card B initializing\n");
    i2c_sw_control.select(I2cSwitch::SecondaryFmc); 
    if (fmc_spi.cpld_init())
      printf("cpld_init failed!\n");
    if (fmc_spi.clocktree_init(clksrc_clktree, vcotype, timing))
      printf("clocktree_init failed!\n");
  }
}

int Module::PrivateData::train_io(unsigned ref_delay)
{
  //
  //  IO Training
  //
  if (!fmca_core.present()) {
    printf("FMC card A not present\n");
    printf("IO training failed!\n");
    return -1;
  }

  bool fmcb_present = fmcb_core.present();

  int rval = -1;
  
  while(1) {

    i2c_sw_control.select(I2cSwitch::PrimaryFmc); 
    if (fmc_spi.adc_enable_test(Flash11))
      break;

    if (fmcb_present) {
      i2c_sw_control.select(I2cSwitch::SecondaryFmc); 
      if (fmc_spi.adc_enable_test(Flash11))
        break;
    }

    //  adcb_core training is driven by adca_core
    adca_core.init_training(0x08);
    if (fmcb_present)
      adcb_core.init_training(ref_delay);

    adca_core.start_training();

    adca_core.dump_training();
  
    if (fmcb_present)
      adcb_core.dump_training();

    i2c_sw_control.select(I2cSwitch::PrimaryFmc); 
    if (fmc_spi.adc_disable_test())
      break;

    if (fmc_spi.adc_enable_test(Flash11))
      break;

    if (fmcb_present) {
      i2c_sw_control.select(I2cSwitch::SecondaryFmc); 
      if (fmc_spi.adc_disable_test())
        break;
      if (fmc_spi.adc_enable_test(Flash11))
        break;
    }

    adca_core.loop_checking();
    if (fmcb_present)
      adcb_core.loop_checking();

    i2c_sw_control.select(I2cSwitch::PrimaryFmc); 
    if (fmc_spi.adc_disable_test())
      break;

    if (fmcb_present) {
      i2c_sw_control.select(I2cSwitch::SecondaryFmc); 
      if (fmc_spi.adc_disable_test())
        break;
    }

    rval = 0;
    break;
  }
  
  return rval;
}

void Module::PrivateData::enable_test_pattern(TestPattern p)
{
  if (p < 8) {
    i2c_sw_control.select(I2cSwitch::PrimaryFmc); 
    fmc_spi.adc_enable_test(p);
    if (fmcb_core.present()) {
      i2c_sw_control.select(I2cSwitch::SecondaryFmc); 
      fmc_spi.adc_enable_test(p);
    }
  }
  else
    base.enableDmaTest(true);
}

void Module::PrivateData::disable_test_pattern()
{
  i2c_sw_control.select(I2cSwitch::PrimaryFmc); 
  fmc_spi.adc_disable_test();
  if (fmcb_core.present()) {
    i2c_sw_control.select(I2cSwitch::SecondaryFmc); 
    fmc_spi.adc_disable_test();
  }
  base.enableDmaTest(false);
}

void Module::PrivateData::enable_cal()
{
  i2c_sw_control.select(I2cSwitch::PrimaryFmc); 
  fmc_spi.adc_enable_cal();
  fmca_core.cal_enable();
  if (fmcb_core.present()) {
    i2c_sw_control.select(I2cSwitch::SecondaryFmc); 
    fmc_spi.adc_enable_cal();
    fmca_core.cal_enable();
  }
}

void Module::PrivateData::disable_cal()
{
  i2c_sw_control.select(I2cSwitch::PrimaryFmc); 
  fmca_core.cal_disable();
  fmc_spi.adc_disable_cal();
  if (fmcb_core.present()) {
    i2c_sw_control.select(I2cSwitch::SecondaryFmc); 
    fmcb_core.cal_disable();
    fmc_spi.adc_disable_cal();
  }
}

void Module::PrivateData::setRxAlignTarget(unsigned t)
{
  unsigned v = gthAlignTarget;
  v &= ~0x3f;
  v |= (t&0x3f);
  gthAlignTarget = v;
}

void Module::PrivateData::setRxResetLength(unsigned len)
{
  unsigned v = gthAlignTarget;
  v &= ~0xf0000;
  v |= (len&0xf)<<16;
  gthAlignTarget = v;
}
 
void Module::PrivateData::dumpRxAlign     () const
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
    
void Module::PrivateData::dumpPgp     () const
{
  string buildStamp = version.buildStamp();
  if (buildStamp.find("pgp")==string::npos)
    return;

  if (buildStamp.find("pgp3")==string::npos) {
    const Pgp2bAxi* pgp = reinterpret_cast<const Pgp2bAxi*>(pgp_reg);
    LPRINT("loopback",_loopback);
    LPRINT("txUserData",_txUserData);
    LPRBF ("rxPhyReady",_status,0,1);
    LPRBF ("txPhyReady",_status,1,1);
    LPRBF ("localLinkReady",_status,2,1);
    LPRBF ("remoteLinkReady",_status,3,1);
    LPRBF ("transmitReady",_status,4,1);
    LPRBF ("rxPolarity",_status,8,3);
    LPRBF ("remotePause",_status,12,0xf);
    LPRBF ("localPause",_status,16,0xf);
    LPRBF ("remoteOvfl",_status,20,0xf);
    LPRBF ("localOvfl",_status,24,0xf);
    LPRINT("remoteData",_remoteUserData);
    LPRINT("cellErrors",_cellErrCount);
    LPRINT("linkDown",_linkDownCount);
    LPRINT("linkErrors",_linkErrCount);
    LPRVC ("remoteOvflVC",_remoteOvfVc);
    LPRINT("framesRxErr",_rxFrameErrs);
    LPRINT("framesRx",_rxFrames);
    LPRVC ("localOvflVC",_localOvfVc);
    LPRINT("framesTxErr",_txFrameErrs);
    LPRINT("framesTx",_txFrames);
    LPRFRQ("rxClkFreq",_rxClkFreq);
    LPRFRQ("txClkFreq",_txClkFreq);
    LPRINT("lastTxOp",_lastTxOpcode);
    LPRINT("lastRxOp",_lastRxOpcode);
    LPRINT("nTxOps",_txOpcodes);
    LPRINT("nRxOps",_rxOpcodes);
  }

  else {
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
    LPRINT("txCntrl"        ,cntrl);
    LPRINT("txStatus"       ,txStatus);
    LPRINT("locStatus"      ,locStatus);
  }

  { printf(" prsnt1L %x\n", (auxStatus>>0)&1);
    printf(" pwrgd1  %x\n", (auxStatus>>1)&1);
    printf(" qsfpPrsN %x\n", (auxStatus>>2)&3);
    printf(" qsfpIntN %x\n", (auxStatus>>4)&3);
    printf(" oe_osc   %x\n", (auxControl>>0)&1);
    printf(" qsfpRstN %x\n", (auxControl>>3)&1);
  }

}

#undef LPRINT
#undef LPRBF
#undef LPRVC
#undef LPRFRQ

void Module::dumpBase() const
{
  p->base.dump();
}

void Module::dumpMap() const
{
  printf("AxiVersion     : %p\n", &p->version);
  printf("FlashController: %p\n", &p->flash);
  printf("I2cSwitch      : %p\n", &p->i2c_sw_control);
  printf("ClkSynth       : %p\n", &p->clksynth);
  printf("LocalCpld      : %p\n", &p->local_cpld);
  printf("FmcSpi         : %p\n", &p->fmc_spi);
  printf("DmaCore        : %p\n", &p->dma_core);
  printf("TprCore        : %p\n", &p->tpr);
  printf("QABase         : %p\n", &p->base);
  printf("HdrFifo        : %p\n", &p->hdr_fifo[0]);
  printf("FexCfg         : %p\n", &p->fex_chan[0]);
}

void Module::PrivateData::setAdcMux(bool     interleave,
                                    unsigned channels)
{
  if (interleave) {
    base.setChannels(channels);
    base.setMode( QABase::Q_ABCD );
    i2c_sw_control.select(I2cSwitch::PrimaryFmc); 
    fmc_spi.setAdcMux(interleave, channels&0x0f);
    if (fmcb_core.present()) {
      i2c_sw_control.select(I2cSwitch::SecondaryFmc); 
      fmc_spi.setAdcMux(interleave, (channels>>4)&0x0f);
    }
  }
  else {
    if (fmcb_core.present()) {
      base.setChannels(channels & 0xff);
      base.setMode( QABase::Q_NONE );
      i2c_sw_control.select(I2cSwitch::PrimaryFmc); 
      fmc_spi.setAdcMux(interleave, (channels>>0)&0xf);
      i2c_sw_control.select(I2cSwitch::SecondaryFmc); 
      fmc_spi.setAdcMux(interleave, (channels>>4)&0xf);
    }
    else {
      base.setChannels(channels & 0xf);
      base.setMode( QABase::Q_NONE );
      i2c_sw_control.select(I2cSwitch::PrimaryFmc); 
      fmc_spi.setAdcMux(interleave, channels&0xf);
    }
  }
}

void Module::init() { GUARD( p->init() ); }

void Module::fmc_init(TimingType timing) { GUARD( p->fmc_init(timing) ); }

void Module::fmc_dump() {
  if (p->fmca_core.present())
    for(unsigned i=0; i<16; i++) {
      p->fmca_core.selectClock(i);
      usleep(100000);
      printf("Clock [%i]: rate %f MHz\n", i, p->fmca_core.clockRate()*1.e-6);
    }
  
  if (p->fmcb_core.present())
    for(unsigned i=0; i<9; i++) {
      p->fmcb_core.selectClock(i);
      usleep(100000);
      printf("Clock [%i]: rate %f MHz\n", i, p->fmcb_core.clockRate()*1.e-6);
    }
}

void Module::fmc_clksynth_setup(TimingType timing)
{
  _sem_i2c.take();
  p->i2c_sw_control.select(I2cSwitch::LocalBus);  // ClkSynth is on local bus
  p->clksynth.setup(timing);
  p->clksynth.dump ();
  _sem_i2c.give();
}

void Module::fmc_modify(int A, int B, int P, int R, int cp, int ab)
{ 
  _sem_i2c.take();
  p->i2c_sw_control.select(I2cSwitch::PrimaryFmc); 
  p->fmc_spi.clocktree_modify(A,B,P,R,cp,ab);
  _sem_i2c.give();
}

uint64_t Module::device_dna() const
{
  uint64_t v = p->version.DeviceDnaHigh;
  v <<= 32;
  v |= p->version.DeviceDnaLow;
  return v;
}


// Update ID advertised on timing link

void Module::set_local_id(unsigned bus)
{
  struct addrinfo hints;
  struct addrinfo* result;

  memset(&hints, 0, sizeof(struct addrinfo));
  hints.ai_family = AF_INET;       /* Allow IPv4 or IPv6 */
  hints.ai_socktype = SOCK_DGRAM; /* Datagram socket */
  hints.ai_flags = AI_PASSIVE;    /* For wildcard IP address */

  char hname[64];
  gethostname(hname,64);
  int s = getaddrinfo(hname, NULL, &hints, &result);
  if (s != 0) {
    fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(s));
    exit(EXIT_FAILURE);
  }

  sockaddr_in* saddr = (sockaddr_in*)result->ai_addr;

  unsigned id = 0xfc000000 | (bus<<16) |
    (ntohl(saddr->sin_addr.s_addr)&0xffff);

  p->base.localId = id;
}

void Module::board_status()
{
  printf("Axi Version [%p]: BuildStamp[%p]: %s\n", 
         &(p->version), &(p->version.BuildStamp[0]), p->version.buildStamp().c_str());

  printf("Dna: %08x%08x  Serial: %08x%08x\n",
         p->version.DeviceDnaHigh,
         p->version.DeviceDnaLow,
         p->version.FdSerialHigh,
         p->version.FdSerialLow );

  _sem_i2c.take();

  p->i2c_sw_control.select(I2cSwitch::LocalBus);
  p->i2c_sw_control.dump();
  
  printf("Local CPLD revision: 0x%x\n", p->local_cpld.revision());
  printf("Local CPLD GAaddr  : 0x%x\n", p->local_cpld.GAaddr  ());
  p->local_cpld.GAaddr(0);

  { unsigned v;
    printf("EEPROM:");
    for(unsigned i=0; i<32; i++) {
      v = p->eeprom[i];
      printf(" %02x", v&0xff);
    }
    printf("\n");
  }

  printf("vtmon1 mfg:dev %x:%x\n", p->vtmon1.manufacturerId(), p->vtmon1.deviceId());
  printf("vtmon2 mfg:dev %x:%x\n", p->vtmon2.manufacturerId(), p->vtmon2.deviceId());
  printf("vtmon3 mfg:dev %x:%x\n", p->vtmon3.manufacturerId(), p->vtmon3.deviceId());

  p->vtmon1.dump();
  p->vtmon2.dump();
  p->vtmon3.dump();

  p->imona.dump();
  p->imonb.dump();

  printf("FMC A [%p]: %s present power %s\n",
         &p->fmca_core,
         p->fmca_core.present() ? "":"not",
         p->fmca_core.powerGood() ? "up":"down");

  printf("FMC B [%p]: %s present power %s\n",
         &p->fmcb_core,
         p->fmcb_core.present() ? "":"not",
         p->fmcb_core.powerGood() ? "up":"down");

  if (p->fmca_core.present()) {
    p->i2c_sw_control.select(I2cSwitch::PrimaryFmc); 
    p->i2c_sw_control.dump();
    printf("vtmona mfg:dev %x:%x\n", p->vtmona.manufacturerId(), p->vtmona.deviceId());
  }

  if (p->fmcb_core.present()) {
    p->i2c_sw_control.select(I2cSwitch::SecondaryFmc); 
    p->i2c_sw_control.dump();
    printf("vtmonb mfg:dev %x:%x\n", p->vtmona.manufacturerId(), p->vtmona.deviceId());
  }

  _sem_i2c.give();
}

void Module::flash_write(const char* fname)
{
  p->flash.write(fname);
}

FlashController& Module::flash() { return p->flash; }

int  Module::train_io(unsigned v) 
{
  int r; GUARD( r = p->train_io(v) ); return r;
}

void Module::enable_test_pattern(TestPattern t) 
{ GUARD( p->enable_test_pattern(t) ); }

void Module::disable_test_pattern() 
{ GUARD( p->disable_test_pattern() ); }

void Module::clear_test_pattern_errors() {
  for(unsigned i=0; i<4; i++) {
    p->fex_chan[i]._test_pattern_errors  = 0;
    p->fex_chan[i]._test_pattern_errbits = 0;
  }
}

void Module::enable_cal () 
{ GUARD( p->enable_cal() ); }

void Module::disable_cal() 
{ GUARD( p->disable_cal()); }

void Module::setAdcMux(unsigned channels)
{
  _sem_i2c.take();
  if (p->fmcb_core.present()) {
    p->base.setChannels(0xff);
    p->base.setMode( QABase::Q_NONE );
    p->i2c_sw_control.select(I2cSwitch::PrimaryFmc); 
    p->fmc_spi.setAdcMux((channels>>0)&0xf);
    p->i2c_sw_control.select(I2cSwitch::SecondaryFmc); 
    p->fmc_spi.setAdcMux((channels>>4)&0xf);
  }
  else {
    p->base.setChannels(0xf);
    p->base.setMode( QABase::Q_NONE );
    p->i2c_sw_control.select(I2cSwitch::PrimaryFmc); 
    p->fmc_spi.setAdcMux(channels&0xf);
  }
  _sem_i2c.give();
}

void Module::setAdcMux(bool     interleave,
                       unsigned channels) 
{ GUARD( p->setAdcMux(interleave, channels) ); }

const Pds::Mmhw::AxiVersion& Module::version() const { return p->version; }
Pds::HSD::TprCore&    Module::tpr    () { return p->tpr; }

void Module::setRxAlignTarget(unsigned v) { p->setRxAlignTarget(v); }
void Module::setRxResetLength(unsigned v) { p->setRxResetLength(v); }
void Module::dumpRxAlign     () const { p->dumpRxAlign(); }
void Module::dumpPgp         () const { p->dumpPgp(); }

void Module::sample_init(unsigned length, 
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

void Module::trig_lcls  (unsigned eventcode)
{
  p->base.setupLCLS(eventcode);
}

void Module::trig_lclsii(unsigned fixedrate)
{
  p->base.setupLCLSII(fixedrate);
}

void Module::trig_daq   (unsigned partition)
{
  p->base.setupDaq(partition);
}

void Module::trig_shift (unsigned shift)
{
  p->base.setTrigShift(shift);
}

void Module::start()
{
  p->base.start();
}

void Module::stop()
{
  p->base.stop();
  p->dma_core.dump();
}

unsigned Module::get_offset(unsigned channel)
{
  _sem_i2c.take();
  p->i2c_sw_control.select((channel&0x4)==0 ? 
                           I2cSwitch::PrimaryFmc :
                           I2cSwitch::SecondaryFmc); 
  unsigned v = p->fmc_spi.get_offset(channel&0x3);
  _sem_i2c.give();
  return v;
}

unsigned Module::get_gain(unsigned channel)
{
  _sem_i2c.take();
  p->i2c_sw_control.select((channel&0x4)==0 ? 
                           I2cSwitch::PrimaryFmc :
                           I2cSwitch::SecondaryFmc); 
  unsigned v = p->fmc_spi.get_gain(channel&0x3);
  _sem_i2c.give();
  return v;
}

void Module::set_offset(unsigned channel, unsigned value)
{
  _sem_i2c.take();
  p->i2c_sw_control.select((channel&0x4)==0 ? 
                           I2cSwitch::PrimaryFmc :
                           I2cSwitch::SecondaryFmc); 
  p->fmc_spi.set_offset(channel&0x3,value);
  _sem_i2c.give();
}

void Module::set_gain(unsigned channel, unsigned value)
{
  _sem_i2c.take();
  p->i2c_sw_control.select((channel&0x4)==0 ? 
                           I2cSwitch::PrimaryFmc :
                           I2cSwitch::SecondaryFmc); 
  p->fmc_spi.set_gain(channel&0x3,value);
  _sem_i2c.give();
}

void Module::clocktree_sync()
{
  _sem_i2c.take();
  p->i2c_sw_control.select(I2cSwitch::PrimaryFmc); 
  p->fmc_spi.clocktree_sync();
  _sem_i2c.give();
}

void Module::sync()
{
  _sem_i2c.take();
  p->i2c_sw_control.select(I2cSwitch::PrimaryFmc); 
  p->fmc_spi.applySync();
  _sem_i2c.give();
}

void* Module::reg() { return (void*)p; }

std::vector<Pgp*> Module::pgp() {
  std::vector<Pgp*> v(0);
  while(1) {
    string buildStamp = p->version.buildStamp();
    if (buildStamp.find("pgp")==string::npos)
      break;
    if (buildStamp.find("pgp3")==string::npos) {
      Pgp2bAxi* pgp = reinterpret_cast<Pgp2bAxi*>(p->pgp_reg);
      for(unsigned i=0; i<4; i++)
        v.push_back(new Pgp2b(pgp[i]));
    }
    else {
      Pgp3Axil* pgp = reinterpret_cast<Pgp3Axil*>(p->pgp_reg);
      for(unsigned i=0; i<4; i++)
        v.push_back(new Pgp3(pgp[i]));
    }
    break;
  }
  return v;
}

Pds::Mmhw::Jtag* Module::xvc() { return &p->xvc; }

FexCfg* Module::fex() { return &p->fex_chan[0]; }

HdrFifo* Module::hdrFifo() { return &p->hdr_fifo[0]; }

uint32_t* Module::trgPhase() { return reinterpret_cast<uint32_t*>(&p->trg_phase[0]); }

void   Module::mon_start()
{
  _sem_i2c.take();
  p->i2c_sw_control.select(I2cSwitch::LocalBus);
  p->vtmon1.start();
  p->vtmon2.start();
  p->vtmon3.start();
  p->imona.start();
  p->imonb.start();
  _sem_i2c.give();
}

EnvMon Module::mon() const
{
  _sem_i2c.take();
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
  _sem_i2c.give();
  return v;
}

void Module::i2c_lock(I2cSwitch::Port v)
{
  _sem_i2c.take(); 
  p->i2c_sw_control.select(v);
}
void Module::i2c_unlock() { _sem_i2c.give(); }
