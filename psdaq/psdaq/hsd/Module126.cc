#include "psdaq/hsd/Module126.hh"
#include "psdaq/hsd/ModuleBase.hh"

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
#include "psdaq/hsd/OptFmc.hh"

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

namespace Pds {
  namespace HSD {
    class Module126::PrivateData {
    public:
      void dumpPgp         () const;
      //
      //  Low level API
      //
    public:
      ModuleBase mbase;          // 0
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
      PhaseMsmt trg_phase;
      uint32_t rsvd_to_0x88000  [(0x5800-4*sizeof(HdrFifo)-sizeof(PhaseMsmt))/4];

      FexCfg   fex_chan[4];      // 0x88000
      uint32_t rsvd_to_0x90000  [(0x8000-4*sizeof(FexCfg))/4];

      //  Pgp (optional)  
      //      Pgp2bAxi pgp[4];           // 0x90000
      //      uint32_t pgp_fmc1;
      //      uint32_t pgp_fmc2;
      uint32_t pgp_reg[0x4000>>2];
      uint32_t rsvd_to_0x98000  [0x4000>>2];
      uint32_t opt_fmc [0x1000>>2]; // 0x98000
    };
  };
};

using namespace Pds::HSD;

Module126* Module126::create(int fd)
{
  void* ptr = mmap(0, sizeof(Pds::HSD::Module126::PrivateData), PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    perror("Failed to map");
    return 0;
  }

  Pds::HSD::Module126* m = new Pds::HSD::Module126;
  m->p = reinterpret_cast<Pds::HSD::Module126::PrivateData*>(ptr);
  m->_fd = fd;

  Pds::Mmhw::RegProxy::initialize(m->p, m->p->mbase.regProxy);

  return m;
}

void Module126::setup_timing()
{
  //
  //  Verify clock synthesizer is setup
  //  Necessary for timing and dual firefly channel pgp
  //
  {
    TprCore& tpr = p->mbase.tpr;
    double txclkr = tpr.txRefClockRate();
    printf("TxRefClk: %f MHz\n", txclkr);

    static const double TXCLKR_MIN = 185.;
    static const double TXCLKR_MAX = 187.;
    if (txclkr < TXCLKR_MIN ||
        txclkr > TXCLKR_MAX) {
      i2c_lock(I2cSwitch::LocalBus);  // ClkSynth is on local bus
      //      i2c().clksynth.setup(LCLSII);
      i2c().clksynth.setup(M3_7);
      i2c_unlock();

      usleep(100000);
      tpr.setLCLSII();
      tpr.resetRxPll();
      usleep(1000000);
      tpr.resetBB();

      p->base.resetFbPLL();
      usleep(1000000);
      p->base.resetFb ();
      p->base.resetDma();
      usleep(1000000);

      tpr.resetCounts();

      optfmc().resetPgp();

      usleep(100000);

      fmc_init(LCLSII);
      if (train_io(0))
        abort();
    }
  }
}

Module126::~Module126()
{
}

int Module126::read(uint32_t* data, unsigned data_size)
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

void Module126::init()
{
  i2c_lock(I2cSwitch::PrimaryFmc); 
  i2c().fmc_spi.initSPI();
  i2c_unlock();
  i2c_lock(I2cSwitch::SecondaryFmc); 
  i2c().fmc_spi.initSPI();
  i2c_unlock();
}

void Module126::fmc_init(TimingType timing)
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

  if (!p->fmca_core.present()) {
    printf("FMC card A not present\n");
    printf("FMC init failed!\n");
    abort();
  }

  {
    printf("FMC card A initializing\n");
    i2c_lock(I2cSwitch::PrimaryFmc); 
    if (i2c().fmc_spi.cpld_init())
      printf("cpld_init failed!\n");
    if (i2c().fmc_spi.clocktree_init(clksrc_clktree, vcotype, timing))
      printf("clocktree_init failed!\n");
    i2c_unlock();
  }

  if (p->fmcb_core.present()) {
    printf("FMC card B initializing\n");
    i2c_lock(I2cSwitch::SecondaryFmc); 
    if (i2c().fmc_spi.cpld_init())
      printf("cpld_init failed!\n");
    if (i2c().fmc_spi.clocktree_init(clksrc_clktree, vcotype, timing))
      printf("clocktree_init failed!\n");
    i2c_unlock();
  }
}

int Module126::train_io(unsigned ref_delay)
{
  //
  //  IO Training
  //
  if (!p->fmca_core.present()) {
    printf("FMC card A not present\n");
    printf("IO training failed!\n");
    return -1;
  }

  bool fmcb_present = p->fmcb_core.present();

  int rval = -1;
  
  while(1) {

    i2c_lock(I2cSwitch::PrimaryFmc); 
    if (i2c().fmc_spi.adc_enable_test(Flash11))
      break;
    i2c_unlock();

    if (fmcb_present) {
      i2c_lock(I2cSwitch::SecondaryFmc); 
      if (i2c().fmc_spi.adc_enable_test(Flash11))
        break;
      i2c_unlock();
    }

    AdcCore& adca_core = p->adca_core;
    AdcCore& adcb_core = p->adcb_core;

    //  adcb_core training is driven by adca_core
    adca_core.init_training(0x08);
    if (fmcb_present)
      adcb_core.init_training(ref_delay);

    adca_core.start_training();

    adca_core.dump_training();
  
    if (fmcb_present)
      adcb_core.dump_training();

    i2c_lock(I2cSwitch::PrimaryFmc); 
    if (i2c().fmc_spi.adc_disable_test())
      break;

    if (i2c().fmc_spi.adc_enable_test(Flash11))
      break;
    i2c_unlock();

    if (fmcb_present) {
      i2c_lock(I2cSwitch::SecondaryFmc); 
      if (i2c().fmc_spi.adc_disable_test())
        break;
      if (i2c().fmc_spi.adc_enable_test(Flash11))
        break;
      i2c_unlock();
    }

    adca_core.loop_checking();
    if (fmcb_present)
      adcb_core.loop_checking();

    i2c_lock(I2cSwitch::PrimaryFmc); 
    if (i2c().fmc_spi.adc_disable_test())
      break;
    i2c_unlock();

    if (fmcb_present) {
      i2c_lock(I2cSwitch::SecondaryFmc); 
      if (i2c().fmc_spi.adc_disable_test())
        break;
      i2c_unlock();
    }

    i2c_lock(I2cSwitch::PrimaryFmc); 
    rval = 0;
    break;
  }
  i2c_unlock();
  
  return rval;
}

void Module126::enable_test_pattern(TestPattern pat)
{
  if (pat < 8) {
    i2c_lock(I2cSwitch::PrimaryFmc); 
    i2c().fmc_spi.adc_enable_test(pat);
    i2c_unlock();
    if (p->fmcb_core.present()) {
      i2c_lock(I2cSwitch::SecondaryFmc); 
      i2c().fmc_spi.adc_enable_test(pat);
      i2c_unlock();
    }
  }
  else
    p->base.enableDmaTest(true);
}

void Module126::disable_test_pattern()
{
  i2c_lock(I2cSwitch::PrimaryFmc); 
  i2c().fmc_spi.adc_disable_test();
  i2c_unlock();
  if (p->fmcb_core.present()) {
    i2c_lock(I2cSwitch::SecondaryFmc); 
    i2c().fmc_spi.adc_disable_test();
    i2c_unlock();
  }
  p->base.enableDmaTest(false);
}

void Module126::enable_cal()
{
  i2c_lock(I2cSwitch::PrimaryFmc); 
  i2c().fmc_spi.adc_enable_cal();
  i2c_unlock();
  p->fmca_core.cal_enable();
  if (p->fmcb_core.present()) {
    i2c_lock(I2cSwitch::SecondaryFmc); 
    i2c().fmc_spi.adc_enable_cal();
    i2c_unlock();
    p->fmcb_core.cal_enable();
  }
}

void Module126::disable_cal()
{
  i2c_lock(I2cSwitch::PrimaryFmc); 
  p->fmca_core.cal_disable();
  i2c().fmc_spi.adc_disable_cal();
  i2c_unlock();
  if (p->fmcb_core.present()) {
    i2c_lock(I2cSwitch::SecondaryFmc); 
    p->fmcb_core.cal_disable();
    i2c().fmc_spi.adc_disable_cal();
    i2c_unlock();
  }
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
    
void Module126::PrivateData::dumpPgp     () const
{
  string buildStamp = mbase.version.buildStamp();
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

  /*
  **  Removed mezzanine card
  { printf(" prsnt1L %x\n", (auxStatus>>0)&1);
    printf(" pwrgd1  %x\n", (auxStatus>>1)&1);
    printf(" qsfpPrsN %x\n", (auxStatus>>2)&3);
    printf(" qsfpIntN %x\n", (auxStatus>>4)&3);
    printf(" oe_osc   %x\n", (auxControl>>0)&1);
    printf(" qsfpRstN %x\n", (auxControl>>3)&1);
  }
  */
}

#undef LPRINT
#undef LPRBF
#undef LPRVC
#undef LPRFRQ

void Module126::dumpBase() const
{
  p->base.dump();
}

void Module126::dumpMap() const
{
  printf("AxiVersion     : %p\n", &p->mbase.version);
  printf("LocalCpld      : %p\n", &i2c().local_cpld);
  printf("FmcSpi         : %p\n", &i2c().fmc_spi);
  printf("DmaCore        : %p\n", &p->mbase.dma_core);
  printf("TprCore        : %p\n", &p->mbase.tpr);
  printf("QABase         : %p\n", &p->base);
  printf("HdrFifo        : %p\n", &p->hdr_fifo[0]);
  printf("FexCfg         : %p\n", &p->fex_chan[0]);
}

void Module126::setAdcMux(bool     interleave,
                          unsigned channels)
{
  if (interleave) {
    p->base.setChannels(channels);
    p->base.setMode( QABase::Q_ABCD );
    i2c_lock(I2cSwitch::PrimaryFmc); 
    i2c().fmc_spi.setAdcMux(interleave, channels&0x0f);
    i2c_unlock();
    if (p->fmcb_core.present()) {
      i2c_lock(I2cSwitch::SecondaryFmc); 
      i2c().fmc_spi.setAdcMux(interleave, (channels>>4)&0x0f);
      i2c_unlock();
    }
  }
  else {
    if (p->fmcb_core.present()) {
      p->base.setChannels(channels & 0xff);
      p->base.setMode( QABase::Q_NONE );
      i2c_lock(I2cSwitch::PrimaryFmc); 
      i2c().fmc_spi.setAdcMux(interleave, (channels>>0)&0xf);
      i2c_unlock();
      i2c_lock(I2cSwitch::SecondaryFmc); 
      i2c().fmc_spi.setAdcMux(interleave, (channels>>4)&0xf);
      i2c_unlock();
    }
    else {
      p->base.setChannels(channels & 0xf);
      p->base.setMode( QABase::Q_NONE );
      i2c_lock(I2cSwitch::PrimaryFmc); 
      i2c().fmc_spi.setAdcMux(interleave, channels&0xf);
      i2c_unlock();
    }
  }
}

void Module126::fmc_dump() {
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

void Module126::fmc_clksynth_setup(TimingType timing)
{
  i2c_lock(I2cSwitch::LocalBus);  // ClkSynth is on local bus
  i2c().clksynth.setup(timing);
  i2c().clksynth.dump ();
  i2c_unlock();
}

void Module126::fmc_modify(int A, int B, int P, int R, int cp, int ab)
{ 
  i2c_lock(I2cSwitch::PrimaryFmc); 
  i2c().fmc_spi.clocktree_modify(A,B,P,R,cp,ab);
  i2c_unlock();
}

uint64_t Module126::device_dna() const
{
  uint64_t v = p->mbase.version.DeviceDnaHigh;
  v <<= 32;
  v |= p->mbase.version.DeviceDnaLow;
  return v;
}


// Update ID advertised on timing link

void Module126::set_local_id(unsigned bus)
{
  unsigned id = ModuleBase::local_id(bus);
  p->base.localId = id;
  p->mbase.version.UserConstants[0] = id;
}

unsigned Module126::remote_id() const { return p->base.partitionAddr; }

void Module126::board_status()
{
  { AxiVersion& v = p->mbase.version;
    printf("Axi Version [%p]: BuildStamp[%p]: %s\n", 
           &v, &v.BuildStamp[0], v.buildStamp().c_str());
    printf("Dna: %08x%08x  Serial: %08x%08x\n",
           v.DeviceDnaHigh,
           v.DeviceDnaLow,
           v.FdSerialHigh,
           v.FdSerialLow ); }

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
  i2c_unlock();

  printf("FMC A [%p]: %s present power %s\n",
         &p->fmca_core,
         p->fmca_core.present() ? "":"not",
         p->fmca_core.powerGood() ? "up":"down");

  printf("FMC B [%p]: %s present power %s\n",
         &p->fmcb_core,
         p->fmcb_core.present() ? "":"not",
         p->fmcb_core.powerGood() ? "up":"down");

  if (p->fmca_core.present()) {
    i2c_lock(I2cSwitch::PrimaryFmc); 
    i2c().i2c_sw_control.dump();
    printf("vtmona mfg:dev %x:%x\n", i2c().vtmona.manufacturerId(), i2c().vtmona.deviceId());
    i2c_unlock();
  }

  if (p->fmcb_core.present()) {
    i2c_lock(I2cSwitch::SecondaryFmc); 
    i2c().i2c_sw_control.dump();
    printf("vtmonb mfg:dev %x:%x\n", i2c().vtmona.manufacturerId(), i2c().vtmona.deviceId());
    i2c_unlock();
  }
}

void Module126::flash_write(const char* fname)
{
  p->mbase.flash.write(fname);
}

FlashController& Module126::flash() { return p->mbase.flash; }

void Module126::clear_test_pattern_errors() {
  for(unsigned i=0; i<4; i++) {
    p->fex_chan[i]._test_pattern_errors  = 0;
    p->fex_chan[i]._test_pattern_errbits = 0;
  }
}

void Module126::setAdcMux(unsigned channels)
{
  _sem_i2c.take();
  if (p->fmcb_core.present()) {
    p->base.setChannels(0xff);
    p->base.setMode( QABase::Q_NONE );
    i2c().i2c_sw_control.select(I2cSwitch::PrimaryFmc); 
    i2c().fmc_spi.setAdcMux((channels>>0)&0xf);
    i2c().i2c_sw_control.select(I2cSwitch::SecondaryFmc); 
    i2c().fmc_spi.setAdcMux((channels>>4)&0xf);
  }
  else {
    p->base.setChannels(0xf);
    p->base.setMode( QABase::Q_NONE );
    i2c().i2c_sw_control.select(I2cSwitch::PrimaryFmc); 
    i2c().fmc_spi.setAdcMux(channels&0xf);
  }
  _sem_i2c.give();
}

const Pds::Mmhw::AxiVersion& Module126::version() const { return p->mbase.version; }
Pds::HSD::TprCore&    Module126::tpr    () { return p->mbase.tpr; }

void Module126::setRxAlignTarget(unsigned v) { p->mbase.setRxAlignTarget(v); }
void Module126::setRxResetLength(unsigned v) { p->mbase.setRxResetLength(v); }
void Module126::dumpRxAlign     () const { p->mbase.dumpRxAlign(); }
void Module126::dumpPgp         () const { p->dumpPgp(); }

void Module126::sample_init(unsigned length, 
                         unsigned delay,
                         unsigned prescale)
{
  p->base.init();
  p->base.samples  = length;
  p->base.prescale = (delay<<6) | (prescale&0x3f);

  p->mbase.dma_core.init(32+48*length);

  p->mbase.dma_core.dump();

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

void Module126::trig_lcls  (unsigned eventcode)
{
  p->base.setupLCLS(eventcode);
}

void Module126::trig_lclsii(unsigned fixedrate)
{
  p->base.setupLCLSII(fixedrate);
}

void Module126::trig_daq   (unsigned partition)
{
  p->base.setupDaq(partition);
}

void Module126::trig_shift (unsigned shift)
{
  p->base.setTrigShift(shift);
}

void Module126::start()
{
  p->base.start();
}

void Module126::stop()
{
  p->base.stop();
  p->mbase.dma_core.dump();
}

unsigned Module126::get_offset(unsigned channel)
{
  i2c_lock((channel&0x4)==0 ? 
           I2cSwitch::PrimaryFmc :
           I2cSwitch::SecondaryFmc); 
  unsigned v = i2c().fmc_spi.get_offset(channel&0x3);
  i2c_unlock();
  return v;
}

unsigned Module126::get_gain(unsigned channel)
{
  i2c_lock((channel&0x4)==0 ? 
           I2cSwitch::PrimaryFmc :
           I2cSwitch::SecondaryFmc); 
  unsigned v = i2c().fmc_spi.get_gain(channel&0x3);
  i2c_unlock();
  return v;
}

void Module126::set_offset(unsigned channel, unsigned value)
{
  i2c_lock((channel&0x4)==0 ? 
           I2cSwitch::PrimaryFmc :
           I2cSwitch::SecondaryFmc); 
  i2c().fmc_spi.set_offset(channel&0x3,value);
  i2c_unlock();
}

void Module126::set_gain(unsigned channel, unsigned value)
{
  i2c_lock((channel&0x4)==0 ? 
           I2cSwitch::PrimaryFmc :
           I2cSwitch::SecondaryFmc); 
  i2c().fmc_spi.set_gain(channel&0x3,value);
  i2c_unlock();
}

void Module126::clocktree_sync()
{
  i2c_lock(I2cSwitch::PrimaryFmc); 
  i2c().fmc_spi.clocktree_sync();
  i2c_unlock();
}

void Module126::sync()
{
  i2c_lock(I2cSwitch::PrimaryFmc); 
  i2c().fmc_spi.applySync();
  i2c_unlock();
}

void* Module126::reg() { return (void*)p; }

std::vector<Pgp*> Module126::pgp() {
  std::vector<Pgp*> v(0);
  while(1) {
    string buildStamp = p->mbase.version.buildStamp();
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

Pds::Mmhw::Jtag* Module126::xvc() { return &p->mbase.xvc; }

FexCfg* Module126::fex() { return &p->fex_chan[0]; }

HdrFifo* Module126::hdrFifo() { return &p->hdr_fifo[0]; }

PhaseMsmt* Module126::trgPhase() { return reinterpret_cast<PhaseMsmt*>(&p->trg_phase); }

OptFmc&     Module126::optfmc() { return *reinterpret_cast<OptFmc*>(p->opt_fmc); }

void   Module126::mon_start()
{
  i2c_lock(I2cSwitch::LocalBus);
  i2c().vtmon1.start();
  i2c().vtmon2.start();
  i2c().vtmon3.start();
  i2c().imona.start();
  i2c().imonb.start();
  i2c_unlock();
}

EnvMon Module126::mon() const
{
  i2c_lock(I2cSwitch::LocalBus);
  EnvMon v;
  Adt7411_Mon m;
  m = i2c().vtmon1.mon();
  v.local12v = m.ain[3]*6.;
  v.edge12v  = m.ain[6]*6.;
  v.aux12v   = m.ain[7]*6.;
  m = i2c().vtmon2.mon();
  v.boardTemp = m.Tint;
  v.local1_8v = m.ain[6];
  m = i2c().vtmon3.mon();
  v.fmc12v = m.ain[2]*6.;
  v.local2_5v = m.ain[6]*2.;
  v.local3_3v = m.ain[7]*2.;

  v.fmcPower   = i2c().imona.power_W();
  v.totalPower = i2c().imonb.power_W();
  i2c_unlock();
  return v;
}

I2c126& Module126::i2c() const
{
  return *reinterpret_cast<I2c126*>(p->mbase.i2c_regs);
}

void Module126::i2c_lock(I2cSwitch::Port port) const
{
  _sem_i2c.take(); 
  const_cast<Module126*>(this)->i2c().i2c_sw_control.select(port);
}
void Module126::i2c_unlock() const { _sem_i2c.give(); }
