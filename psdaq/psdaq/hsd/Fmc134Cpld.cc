#include "psdaq/hsd/Fmc134Cpld.hh"
#include <unistd.h>
#include <stdio.h>

using namespace Pds::HSD;

enum { C0_EXT = 4, C0_LMK = 8, 
       C0_ADC_0 = 0x40, C0_ADC_1 = 0x80 };
enum { C0_SYNC_EXT = 0, C0_SYNC_FPGA = 0x10, C0_SYNC_0 = 0x20, C0_SYNC_1 = 0x30 };

enum { C1_OSC = 1, C1_REF = 2, C1_AD = 4, C1_LMX = 8, C1_FF_CS = 0x10,
       C1_FF_RST = 0x20, C1_LED = 0x40, C1_EEPROM = 0x80 };

void Fmc134Cpld::initialize(bool lDualChannel,
                            bool lInternalRef)
{
  unsigned v;

  if (lInternalRef) {
    v = C1_FF_CS | C1_FF_RST |  // firefly intf active low 
      C1_AD | C1_OSC | C1_REF | C1_LMX | 
      C1_EEPROM;  // bit 7 reserved
    _control1 = v;

    v = C0_SYNC_FPGA;
    _control0 = v; 
  }
  else {
    // turn off 0sc, point at ext ref enable LMX bits 3, 1, and 0
    v = C1_AD | C1_LMX | C1_LED | C1_EEPROM;
    _control1 = v;
    
    v = C0_SYNC_FPGA | C0_EXT;
    _control0 = v;
  }

  usleep(100000);

  _lmx_init(lInternalRef);
  lmx_dump();

  _hmc_init();

  //  Reset LMK
  v = _control0 | C0_LMK;
  _control0 = v;
  _control0 = v & ~C0_LMK;
  usleep(5000);
  
  _lmk_init();

  //  Reset ADCs
  v = _control1;
  v &= ~C1_AD;
  _control1 = v;
  v |= C1_AD;
  _control1 = v;
  usleep(2000);

  _adc_init(0, lDualChannel);
  _adc_init(1, lDualChannel);
}

void Fmc134Cpld::enable_mon(bool v)
{
  v = _control1;
  if (v) {
    _control1 = v | C1_AD;  // clear AD7291 reset
    usleep(10000);
  }
  else
    _control1 = v & ~C1_AD; // reassert AD7291 reset
}

void Fmc134Cpld::enable_adc_prbs(bool v)
{
  writeRegister(LMK, 0x205, v ? 3:0);
}

void Fmc134Cpld::_lmx_init(bool lInternalRef)
{
#if 0
#define WRREG(r,v) {                                             \
    printf("--\n");                                              \
    printf("LMX read (%u): %08x\n", r, readRegister(LMX,r));     \
    writeRegister(LMX, r, v);                                    \
    printf("LMX write(%u): %08x\n", r, v);                       \
    printf("LMX read (%u): %08x\n", r, readRegister(LMX,r));     \
  }
#else
#define WRREG(r,v) writeRegister(LMX,r,v)
#endif

  // R5.4 = 1
  WRREG( 5, 0x4087001);  // lowest bit is reset (why do the rest matter?)

  // R15, R13, R10, R9, R8, R7, R6, R5, R4, R3, R2, R1, R0
  WRREG( 13, 0x4081C10);  // DLD TOL 6 ns
  WRREG( 10, 0x0210050C); // default correction
  WRREG(  9, 0x003C7C03);
  WRREG(  8, 0x0207DDBF);
  // WRREG(  7, ((0<<22) | (0<<19) | (0<<18) |  // FL_SELECT=GND, FL_PINMODE=0, FL_INV=0
  //             (4<<13) | (1<< 9) | (0<<12) |  // MUXOUT_SELECT=READBACK
  //             (0<< 4) | (0<< 0) | (0<< 3))); // LD_SELECT=GND
  WRREG(  7, 0x0004E211); // 4DSP value
  unsigned v;
  if (lInternalRef) {
    WRREG(  6, 0x4C);
    WRREG(  5, 0x0030808);
    WRREG(  4, 0);
    WRREG(  3, 0x20040BE);
    WRREG(  2, 0x0FD0902);
    WRREG(  1, 0xF800001);
    v = 0x06020000; // N = 32
  }
  else {
    WRREG(  6, 0x4C);
    WRREG(  5, 0x6010800); // Disable BUF_EN pin, Bypass VCO divider, Use VCO_SEL VCO to start 
    WRREG(  5, 0x0010800); // Disable BUF_EN pin, Bypass VCO divider, Use VCO_SEL VCO to start 
    WRREG(  4, 0);
    WRREG(  3, 0x20000BE);
    WRREG(  2, (3<<22)); // CPP negative, No fractional divider
    WRREG(  1, 0xFC00001);
    v = (0<<27) | (3<<25) | (256<<12); // ID=0, Frac_dither=Disabled, N=256
  }

  lmx_dump();
  WRREG(  0, v );
  // wait 300 ms
  usleep(300000);
  // R0 again
  WRREG(  0, v );

#undef WRREG
}

void Fmc134Cpld::_lmk_init()
{
#define RDREG(r) { printf("LMK read (0x%04x): %08x\n", r, readRegister(LMK,r)); }
#define WRREG(r,v) {                                     \
    writeRegister(LMK, r, v);                            \
    unsigned q = readRegister(LMK,r);                    \
    if (q!=v) {                                          \
      printf("LMK write(0x%04x): %08x [%08x]\n", r,v,q); \
    }                                                    \
}

  writeRegister(LMK,0,0x80);  // reset
  writeRegister(LMK,0,0);     // clear reset
  writeRegister(LMK,0,0x10);  // force to 4-wire
  WRREG(0x148, 0x33);
  WRREG(2,0);     // clear powerdown

  RDREG(3);
  RDREG(4);
  RDREG(5);
  RDREG(0xc);
  RDREG(0xd);

  // DCLKout0, SDCLKout1  (GBTCLK0,1 M2C LVDS @320MHz)
  WRREG(0x100, 0x0a);  // DCLKout0_DIV=10, IDL,ODL=0
  WRREG(0x101, 0x00);  // DCLKout0_DDLY_CNTH/L
  WRREG(0x102, 0x70);  // 
  WRREG(0x103, 0x40);
  WRREG(0x104, 0);     // SDCLKout1
  WRREG(0x105, 0);     // SDCLKout1
  WRREG(0x106, 0);    
  WRREG(0x107, 0x11);  // DCLKout0,SDCLKout1 LVDS

  // DCLKout2, SDCLKout3 (160MHz)
  WRREG(0x108, 0x14);  // DCLKout2_DIV=20, IDL,ODL=0
  WRREG(0x109, 0x00);  // DCLKout2_DDLY_CNTH/L
  WRREG(0x10a, 0x70);
  WRREG(0x10b, 0x40);
  WRREG(0x10c, 0x20);  // SDCLKout3
  WRREG(0x10d, 0);     // SDCLKout3
  WRREG(0x10e, 0);   
  WRREG(0x10f, 0x11);  // DCLKout2,SDCLKout3 LVDS

  // DCLKout4, SDCLKout5
  WRREG(0x110, 0x20);  // DCLKout4_DIV=20, IDL,ODL=0
  WRREG(0x111, 0x00);  // DCLKout4_DDLY_CNTH/L=10 (clks high,low)
  WRREG(0x112, 0x70);  // Divider only
  WRREG(0x113, 0x40);  // Divider only
  WRREG(0x114, 0x20);  // SDCLKout5
  WRREG(0x115, 0);     // SDCLKout5
  WRREG(0x116, 0);   
  WRREG(0x117, 0x60);

  // DCLKout6, SDCLKout7
  WRREG(0x118, 0x02);
  WRREG(0x119, 0x00);
  WRREG(0x11a, 0x70);
  WRREG(0x11b, 0x48);
  WRREG(0x11c, 0x30);
  WRREG(0x11d, 0);     // SDCLKout3
  WRREG(0x11e, 0);     // powerdown clock group
  WRREG(0x11f, 0);     // DCLKout pwdn,SDCLKout pwdn

  // DCLKout8, SDCLKout9
  WRREG(0x120, 0x02);
  WRREG(0x121, 0x00);
  WRREG(0x122, 0x70);
  WRREG(0x123, 0x48);  // Divider only
  WRREG(0x124, 0x30);
  WRREG(0x125, 0);     // SDCLKout5
  WRREG(0x126, 0);     // powerdown DCLKout4 analog delay
  WRREG(0x127, 0);     // DCLKout pwdn,SDCLKout pwdn

  // DCLKout10, SDCLKout11
  WRREG(0x128, 0x20);  // DCLKout2_DIV=20, IDL,ODL=0
  WRREG(0x129, 0x00);  // DCLKout2_DDLY_CNTH/L=10 (clks high,low)
  WRREG(0x12a, 0x70);  // Divider only
  WRREG(0x12b, 0x40);  // Divider only
  WRREG(0x12c, 0x20);  // SDCLKout3
  WRREG(0x12d, 0);     // SDCLKout3
  WRREG(0x12e, 0);     // powerdown DCLKout2 analog delay
  WRREG(0x12f, 0x60);  // DCLKout10 pwdn,SDCLKout11 LVPECL

  // DCLKout12, SDCLKout13
  WRREG(0x130, 0x0a);  // DCLKout0_DIV=20, IDL,ODL=0
  WRREG(0x131, 0);
  WRREG(0x131, 0x70);  // DCLKout0_DDLY_CNTH/L=10 (clks high,low)
  WRREG(0x133, 0x40);  // Divider only
  WRREG(0x134, 0);     // SDCLKout1
  WRREG(0x135, 0);     // SDCLKout1
  WRREG(0x136, 0);     // powerdown DCLKout0 analog delay
  WRREG(0x137, 0x11);  // DCLKout0,SDCLKout1 LVDS

  WRREG(0x138, 0x40);
  WRREG(0x139, 0x03);  // SYSREF_MUX SYSREF_Free_Running (really want SYNC input - revisit)
  WRREG(0x13A, 0x01);  // SYSREF divider MSB (10MHz)
  WRREG(0x13B, 0x40);  // SYSREF divider LSB
  WRREG(0x13C, 0);     // SYSREF delay MSB (unused)
  WRREG(0x13D, 0x08);  // SYSREF delay LSB
  WRREG(0x13E, 0);     // SYSREF_PULSE_CNT=1
  WRREG(0x13F, 0);     // 
  WRREG(0x140, 0xf1);  // powerdown PLL1,VCO, SYSREF_PULSER
  WRREG(0x141, 0x00);  // Disable all dynamic digital delays
  WRREG(0x142, 0x00);  // 
  WRREG(0x143, 0x70);  // (Enable SYNC, edge-sensitive - revisit)
  WRREG(0x144, 0xff);
  WRREG(0x145, 0x0);
  WRREG(0x146, 0);

  WRREG(0x147, 0x10);  // internal ref only
  WRREG(0x148, 0x33);
  WRREG(0x149, 0x00);
  WRREG(0x14A, 0x00);
  WRREG(0x14B, 0x02);
  WRREG(0x14C, 0x00);
  WRREG(0x14D, 0x00);
  WRREG(0x14E, 0x00);
  WRREG(0x14F, 0x7F);
  WRREG(0x150, 0x00);
  WRREG(0x151, 0x02);
  WRREG(0x152, 0x00);
  WRREG(0x153, 0x00);
  WRREG(0x154, 0x80);
  WRREG(0x155, 0x00);
  WRREG(0x156, 0x80);
  WRREG(0x157, 0x03);
  WRREG(0x158, 0xE8);
  WRREG(0x159, 0x00);
  WRREG(0x15A, 0x05);
  WRREG(0x15B, 0xF4);
  WRREG(0x15C, 0x20);
  WRREG(0x15D, 0x00);
  WRREG(0x15E, 0x00);
  WRREG(0x15F, 0x03);
  WRREG(0x160, 0x00);
  WRREG(0x161, 0x04);
  WRREG(0x162, 0xCC);
  WRREG(0x163, 0x00);
  WRREG(0x164, 0x00);
  WRREG(0x165, 0x04);

  WRREG(0x145, 0);
  WRREG(0x171, 0xaa);
  WRREG(0x172, 0x02);
  WRREG(0x17c, 0x15);
  WRREG(0x17d, 0x33);

  WRREG(0x166, 0x00);
  WRREG(0x167, 0x00);
  WRREG(0x168, 0x04);
  WRREG(0x169, 0x49);
  WRREG(0x16A, 0x00);
  WRREG(0x16B, 0x20);
  WRREG(0x16C, 0x00);
  WRREG(0x16D, 0x00);
  WRREG(0x16E, 0x12);
  WRREG(0x173, 0x60);  // Powerdown PLL2

  usleep(100000);

  WRREG(0x183,1);
  WRREG(0x183,0);

#if 1
        // try to sync all the output dividers
        // SYNC_MODE enable to SYNC event
        // SYSREF_CLR = 1
        // SYNC_1SHOT_EN = 1
        // SYNC_POL = 0 (Normal)
        // SYNC_EN = 1
        // SYNC_MODE = 1 (sync_event_generated from SYNC pin)
  WRREG(0x143,0xD1);
        // change SYSREF_MUX to normal SYNC (0)
  WRREG(0x139,0x00);
        // Enable dividers reset
  WRREG(0x144,0x00);
        //toggle the polarity (keep SYSREF_CLR active)
  WRREG(0x143,0xF1);

  usleep(10000);

  WRREG(0x143,0xD1);
        // disable dividers
  WRREG(0x144,0xFF); 
        // change SYSREF_MUX back to continuous
  WRREG(0x139,0x03);
        // restore SYNC_MODE & remove SYSREF_CLR
  WRREG(0x143,0x50);
#else
  
#endif

#undef RDREG
#undef WRREG
}

void Fmc134Cpld::_adc_init(unsigned a, bool ldualch)
{
#define RDREG(r) { printf("ADC read (0x%04x): %08x\n", r, readRegister(dev,r)); }
#if 0
#define WRREG(r,v) { \
    printf("..\n");                           \
    RDREG(r);                                 \
    writeRegister(dev, r, v);                 \
    printf("ADC write(0x%04x): %08x\n", r,v); \
    RDREG(r); }
#else
#define WRREG(r,v) {                                            \
    writeRegister(dev,r,v);                                     \
    unsigned q = readRegister(dev,r);                           \
    if (q!=v) {                                                 \
      printf("ADC wrote(0x%04x): %08x[%08x]\n", r, v, q);       \
    }                                                           \
  }
#endif
  
  DevSel dev = (a==0) ? ADC0 : ADC1;
  
  printf("--ADC%u--\n",a);
  RDREG(3);
  RDREG(4);
  RDREG(5);
  RDREG(6);
  RDREG(0xc);
  RDREG(0xd);

  WRREG(0x00,0xB0);  // reset
  usleep(10);

        // Set the D Clock  and SYSREF input pins to LVPECL
  WRREG(0x2A,0x06);
        // Set Timestamp input pins to LVPECL but do not enable timestamp
  WRREG(0x3B,0x02);
        // Invert ADC0 Clock            (write to only ADC0)
  if (a==0) {
    WRREG(0x2B7,0x01);
  }

        // Enable SYSREF Processor
  WRREG(0x29,0x20);
  WRREG(0x29,0x60);

  // Procedure:
  //  Reset
  //  Appy stable device CLK
  //  Program JESD_EN=0
  WRREG(0x200,0);
  //  Program CAL_EN=0
  WRREG(0x61,0);

        // Enable SYSREF Calibration while background calibration is disabled
        // Set 256 averages with 256 cycles per accumulation
  WRREG(0x2B1,0x0F);
        // Start SYSREF Calibration (~0.1 sec)
  WRREG(0x2B0, 0x01);
  usleep(500000);

  unsigned v = readRegister(dev,0x2B4);
  if (v&2)
    printf("ADC SYSREF Calibration Done\n");
  else
    printf("ADC SYSREF Calibration Not Done\n");

  // Set CAL_BG to enable background calibration
  WRREG(0x62, 0x02);

  //  Program JMODE
  WRREG(0x201,ldualch ? 2:0);
  //  Program KM1 (K-1)
  WRREG(0x202,0xf);
        // Keep output format as 2's complement and ENABLE Scrambler
  WRREG(0x204, 0x03);
  //  Program SYNC_SEL (choose inputs) - default OK
  //  Configure calibration settings, fg/bg mode, offset
  //  Program CAL_EN=1
  WRREG(0x61,1);
  //  Enable overrange OVR_EN
  //  WRREG(0x213,0xf);
  //  Program JESD_EN=1
  WRREG(0x200,1);

  WRREG(0x30,0xff);  // Set analog full scale to 950mVpp
  WRREG(0x31,0xff); 
  WRREG(0x32,0xff);  // Set analog full scale to 950mVpp
  WRREG(0x33,0xff); 
  usleep(5000);

  unsigned adc_txemphasis = 0;
        // Configure the transceiver pre-emphasis setting (0 to 0xF)
  WRREG(0x048, adc_txemphasis);

  // Disable SYSREF Processor in ADC before turning off SYSREF outputs
  WRREG(0x029, 0x00);
  // Disable SYSREF to ADC
  writeRegister(LMK, (a==0) ? 0x12F:0x117, 0x00);
  // ( Alternately, keep SYSREF running and track with Tad )

  v = readRegister(dev,0x208);  // JESD Status [7:0]=[RSVD,LINKUP,SYNC,REALIGN,ALIGNED,LOCKD,RSVD,RSVD]
  printf("JESD STATUS:");
  printf(" %s", v&(1<<6) ? "LinkUp":"LinkDn");
  printf(" %s", v&(1<<5) ? "Sync":"nSync");
  if (v&(1<<4)) printf(" Realign");
  if (v&(1<<3)) printf(" Align");
  printf(" %s\n", v&(1<<2) ? "Locked":"notLocked");

#undef RDREG
#undef WRREG
}

void Fmc134Cpld::_hmc_init()
{
#define RDREG(r) { printf("HMC read (%u): %04x\n", r, readRegister(HMC,r)); }
#define WRREG(r,v) { writeRegister(HMC,r,v); }

  WRREG(0,1);  // reset
  WRREG(1,1);  // chip enable
  WRREG(2,0x91); // enable buffers
  WRREG(3,0x1a);
  WRREG(4,0);
  WRREG(5,0x3a);

#undef RDREG
#undef WRREG
}

void Fmc134Cpld::dump() const
{
  static const char* sync_src[] = {"EXT_TRIG","FPGA_TRIG","0","1"};
  unsigned v = _control0;
  printf("EXT_SAMPLE_CLK %u\n", (v & C0_EXT) ? 1:0);
  printf("LMK04832_RESET %u\n", (v & C0_LMK) ? 1:0);
  printf("SYNC_SRC_SEL   %s\n", sync_src[(v>>4)&3]);
  printf("ADC0_CALTRIG   %u\n", (v & C0_ADC_0) ? 1:0);
  printf("ADC1_CALTRIG   %u\n", (v & C0_ADC_1) ? 1:0);

  v = _control1;
  printf("OSC_100_EN     %u\n", (v & C1_OSC) ? 1:0);
  printf("REF_INT        %u\n", (v & C1_REF) ? 1:0);
  printf("AD7291_RST_L   %u\n", (v & C1_AD) ? 1:0);
  printf("LMX2581_EN     %u\n", (v & C1_LMX) ? 1:0);
  printf("FF_CS_L        %u\n", (v & C1_FF_CS) ? 1:0);
  printf("FF_RST_L       %u\n", (v & C1_FF_RST) ? 1:0);
  printf("LED_EN         %u\n", (v & C1_LED) ? 1:0);
  printf("EEPROM_WP      %u\n", (v & C1_EEPROM) ? 1:0);

  printf("CONTROL : %02x %02x\n", _control0&0xff, _control1&0xff);
  unsigned q = _status;
  printf("ADC0_CAL_STAT ALARM %c\n", (q&1)?'T':'F');
  printf("ADC1_CAL_STAT ALARM %c\n", (q&2)?'T':'F');
  printf("FMC134 TEMP ALARM   %c\n", (q&4)?'T':'F');
  printf("FMC134 VOLT ALARM   %c\n", (q&8)?'T':'F');
  printf("LMX LOCK DETECT     %c\n", (q&0x10)?'T':'F');
  if (~q&0x20)  printf("FF_PRSNT\n");
  printf("GA_MOD              %u\n", ((q>>6)&3));
}

void Fmc134Cpld::lmx_dump() 
{
  unsigned v = const_cast<Fmc134Cpld*>(this)->readRegister(LMX,6);
  printf("LMX Status:\n");
  printf("  BUFEN     %u\n",(v>>0)&1);
  printf("  CE        %u\n",(v>>1)&1);
  printf("  LD        %u\n",(v>>2)&1);
  printf("  DLD       %u\n",(v>>3)&1);
  printf("  FLOUT_L   %u\n",(v>>4)&1);
  printf("  VCO_VALID %u\n",(v>>5)&1);
  printf("  VCO_TUNEH %u\n",(v>>6)&1);
  printf("  VCO_RAILL %u\n",(v>>8)&1);
  printf("  VCO_RAILH %u\n",(v>>9)&1);
  printf("  CAL_RUN   %u\n",(v>>10)&1);
  printf("  VCO_DET   %u\n",(v>>15)&1);
  printf("  OSC_DET   %u\n",(v>>16)&1);
  printf("  FIN_DET   %u\n",(v>>17)&1);
  printf("  VCO_SEL   %u\n",(v>>18)&3);
}

void Fmc134Cpld::lmk_dump() 
{
#define RDSTAT(reg,ttl,dfl) {                   \
    unsigned v = readRegister(LMK,reg);         \
    printf("%9.9s: 0x%x [%x]\n",v,dfl); }

#define RDSTAT2(reg,ttl,dfl) {                   \
    unsigned v = readRegister(LMK,reg);          \
    v |= (readRegister(LMK,reg+1)<<8);           \
    printf("%9.9s: 0x%x [%x]\n",v,dfl); }

  DevSel dev = LMK;

  printf("--LMK--\n");
  RDSTAT (3,DevType,0);
  RDSTAT2(4,Prod   ,0);
  RDSTAT (6,Rev    ,0);
  RDSTAT2(12,Vendor,0);

#undef RDSTAT  
#undef RDSTAT2
}

void Fmc134Cpld::adc_dump(unsigned a)
{
  DevSel dev = (a==0) ? ADC0 : ADC1;
  printf("--ADC%u--\n",a);
  
#define RDSTAT(reg,ttl,dfl) {                   \
    unsigned v = readRegister(dev,reg);         \
    printf("%9.9s: 0x%x [%x]\n",#ttl,v,dfl); }

#define RDSTAT2(reg,ttl,dfl) {                  \
    unsigned v = readRegister(dev,reg);         \
    v |= (readRegister(dev,reg+1)<<8);          \
    printf("%9.9s: 0x%x [%x]\n",#ttl,v,dfl); }

#define RDSTAT3(reg,ttl,dfl) {                  \
    unsigned v = readRegister(dev,reg);         \
    v |= (readRegister(dev,reg+1)<<8);          \
    v |= (readRegister(dev,reg+2)<<16);         \
    printf("%9.9s: 0x%x [%x]\n",#ttl,v,dfl); }
  
  RDSTAT ( 3,ChipType,0x03);
  RDSTAT2( 4,ChipID  ,0x20);
  RDSTAT ( 6,ChipVsn ,0x0a);
  RDSTAT2(12,VendorId,0x451);

  RDSTAT (0x200,JESDEn ,0);
  RDSTAT (0x201,JMode  ,0);
  RDSTAT (0x202,K      ,0);
  {
    unsigned v = readRegister(dev,0x208);
    printf("JESD STATUS:");
    printf(" %s", v&(1<<6) ? "LinkUp":"LinkDn");
    printf(" %s", v&(1<<5) ? "Sync":"nSync");
    if (v&(1<<4)) printf(" Realign");
    if (v&(1<<3)) printf(" Align");
    printf(" %s\n", v&(1<<2) ? "Locked":"notLocked");
  }

  RDSTAT3(0x2b2,SysRefSt,0);
  RDSTAT3(0x2b5,SysRefTad,0);

  RDSTAT3(0x2c,SysRefPos,0);

#undef RDSTAT  
#undef RDSTAT2
#undef RDSTAT3
}

void Fmc134Cpld::writeRegister(DevSel   dev,
                               unsigned address,
                               unsigned data)
{
#if 0
  uint32_t v;
  switch(dev) {
  case LMK:
    v = ((address&0x1fff)<<16) | ((data&0xff)<<8);
    break;
  case LMX:
    v = ((data&0xfffffff)<<4) | ((address&0xf)<<0);
    break;
  case HMC:
    v = ((data&0x1ff)<<23) | ((address&0xf)<<19) | (1<<16);
    break;
  default:
    v = ((address&0x7fff)<<16) | ((data&0xff)<<8);
    break;
  }
 
  for(unsigned i=0; i<4; i++) {
    _i2c_data[i] = v&0xff; 
    v>>=8;
  }
  _command = (1<<dev);

  v = _i2c_read[1];
  v = _i2c_read[2];
  v = _i2c_read[3];
#else
  switch(dev) {
  case LMK:
    _i2c_data[3] = (address>>8)&0x1f;
    _i2c_data[2] = (address>>0)&0xff;
    _i2c_data[1] = data&0xff;
    _command = 1<<dev;
    break;
  case LMX:
    _i2c_data[3] = (data>>20)&0xff;
    _i2c_data[2] = (data>>12)&0xff;
    _i2c_data[1] = (data>> 4)&0xff;
    _i2c_data[0] = ((data&0xf)<<4) | (address&0xf);
    _command = 1<<dev;
    break;
  case HMC:
    _i2c_data[3] = (data>>1)&0xff;
    _i2c_data[2] = ((data&1)<<7) | ((address&0xf)<<3) | 1;
    _command = 1<<dev;
    break;
  default:
    _i2c_data[3] = (address>>8)&0x7f;
    _i2c_data[2] = (address>>0)&0xff;
    _i2c_data[1] = (data&0xff);
    _command = 1<<dev;
    break;
  }
#endif
}

unsigned Fmc134Cpld::readRegister(DevSel   dev,
                                  unsigned address)
{
  unsigned data = -1U;
#if 0
  uint32_t v;
  switch(dev) {
  case LMK:
    v = (1<<31) | ((address&0x1fff)<<16);
    break;
  case LMX:
    v = (1<<10) | ((address&0xf)<<5) | 6;
    break;
  case HMC:
    return -1;
  default:
    v = (1<<31) | ((address&0x7fff)<<16);
    break;
  }

  for(unsigned i=0; i<4; i++) {
    _i2c_data[i] = v&0xff; 
    v>>=8;
  }

  usleep(10000);

  _command = (1<<dev);
  usleep(10000);
  
  if (dev == LMX) {
    _command = (1<<dev); 
    usleep(10000);
    data = (_read()>>4);
  }
  else {
    data = (_i2c_read[1]&0xff);
  }
#else
  switch(dev) {
  case LMK:
    _i2c_data[3] = ((address>>8)&0x1f) | 0x80;
    _i2c_data[2] = ((address>>0)&0xff);
    _i2c_data[1] = 0xff;
    _command = 1<<dev;
    data = (_read()>>8)&0xff;
    break;
  case LMX:
    _i2c_data[3] = (data>>20)&0x7f;
    _i2c_data[2] = (data>>12)&0xff;
    _i2c_data[1] = ((address>>3)&1) | (1<<2);
    _i2c_data[0] = ((address&0x7)<<5) | 6;
    _command = 1<<dev;

    _i2c_data[3] = (data>>20)&0x7f;
    _i2c_data[2] = (data>>12)&0xff;
    _i2c_data[1] = ((address>>3)&1) | (1<<2);
    _i2c_data[0] = ((address&0x7)<<5) | 6;
    _command = 1<<dev;

    data = _read()>>4;
    break;
  case HMC:
    return -1;
  default:
    _i2c_data[3] = ((address>>8)&0x7f) | 0x80;
    _i2c_data[2] = (address>>0)&0xff;
    _i2c_data[1] = (data&0xff);
    _command = 1<<dev;
    data = (_read()>>8)&0xff;
    break;
  }
#endif

  return data;
}

unsigned Fmc134Cpld::_read()
{ return
    ((_i2c_read[0]&0xff)<< 0) |
    ((_i2c_read[1]&0xff)<< 8) |
    ((_i2c_read[2]&0xff)<<16) |
    ((_i2c_read[3]&0xff)<<24); }


