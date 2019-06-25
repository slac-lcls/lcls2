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
  //  WRREG(  7, 0x0004E211); // 4DSP value  {LD_PINMODE=1, LD_INV=0, LD_SEL=1, MUXOUT_PINMODE=1, MUX_INV=0, MUXOUT_SEL=7, FL_INV=1, FL_PINMODE=0, FL_SEL=0}
  WRREG(  7, 0x00048211);  // MUXOUT_SEL=4
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
  WRREG(0x201,(ldualch ? 2:0));
  //  Program KM1 (K-1)
  WRREG(0x202,0xf);
  // Reg 0x204: b0 - Enable scrambler      / disable scrambler
  //            b1 - 2's complement format / offset binary
  // Keep output format as 2's complement and ENABLE Scrambler
  //  WRREG(0x204, 0x03);  // 
  //  WRREG(0x204, 0x02);  // Disable scrambler
  //  WRREG(0x204, 0x00);  // Disable scrambler, output format is offset binary
  WRREG(0x204, 0x01);  // Enable scrambler, output format is offset binary
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
  printf("--LMX--\n");
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
    printf("%9.9s: 0x%x [%x]\n",#ttl,v,dfl); }

  //  Reset clock chip
  { unsigned v = _control0;
    _control0 = v | (1<<3);
    usleep(1);
    _control0 = v;
    usleep(5000); }

  writeRegister(LMK,0,0x80);  // Force a reset
  usleep(1);
  writeRegister(LMK,0,0);     // Clear reset
  usleep(1);
  writeRegister(LMK,0,0x10);  // Force SPI to 4-Wire
  
  printf("--LMK--\n");
  RDSTAT (3,DevType,0x06);
  RDSTAT (4,ProdMSB,0xd1);
  RDSTAT (5,ProdLSB,0x63);
  RDSTAT (6,Rev    ,0x70);
  RDSTAT (12,Vendor,0x51);
  RDSTAT (13,Vendor,0x04);

#undef RDSTAT  
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
                               unsigned value)
{
  unsigned data=0;
  switch(dev) {
  case LMK:
    data |= (address & 0x1fff) << 16;
    data |= (value & 0xff) << 8;
    break;
  case LMX:
    data |= (address & 0xf);
    data |= (value & 0xfffffff) <<4;
    break;
  case HMC:
    data |= (1<<16);
    data |= (address & 0xf) << 19;
    data |= (value & 0x1ff) << 23;
    break;
  default:
    data |= (address&0x7fff) << 16;
    data |= (value & 0xff) << 8;
    break;
  }

  _i2c_data[0] = data>>0;
  _i2c_data[1] = data>>8;
  _i2c_data[2] = data>>16;
  _i2c_data[3] = data>>24;

  _command = dev;
  usleep(10000);

  volatile unsigned v __attribute__((unused));
  v = (_i2c_read[1]<<8) | (_i2c_read[2]<<16) | (_i2c_read[3]<<24);
}

unsigned Fmc134Cpld::readRegister(DevSel   dev,
                                  unsigned address)
{
  unsigned data = 0;
  switch(dev) {
  case LMK:
    data |= (1<<31);
    data |= (address&0x1fff)<<16;
    break;
  case LMX:
    data |= (address&0xf)<<5;
    data |= (1<<10);
    data |= 6;
    break;
  case HMC:
    return -1;
  default:
    data |= (1<<31);
    data |= (address&0x7fff)<<16;
    break;
  }

  _i2c_data[0] = data>>0;
  _i2c_data[1] = data>>8;
  _i2c_data[2] = data>>16;
  _i2c_data[3] = data>>24;

  _command = dev;
  usleep(10000);

  if (dev==LMX) {
    _command = dev;
    usleep(10000);

    data = _read();
  }
  else {
    data = _i2c_read[1]&0xff;
  }

  return data;
}

unsigned Fmc134Cpld::_read()
{ return
    ((_i2c_read[0]&0xff)<< 0) |
    ((_i2c_read[1]&0xff)<< 8) |
    ((_i2c_read[2]&0xff)<<16) |
    ((_i2c_read[3]&0xff)<<24); }


#define LMK_SELECT LMK
#define LMX_SELECT LMX
#define HMC_SELECT HMC
#define ADC_SELECT_BOTH ADC_BOTH
#define ADC0_SELECT ADC0
#define ADC1_SELECT ADC1
#define cpld_address 0
#define spi_write(unit,sel,reg,val) writeRegister(sel,reg,val)
#define spi_read(unit,sel,reg,valp) *(valp) = readRegister(sel,reg)
#define i2c_write(unit,addr,val)  reinterpret_cast<uint32_t*>(this)[addr]=val
#define i2c_read(unit,addr,valp)  *(valp) = reinterpret_cast<uint32_t*>(this)[addr]
#define unitapi_sleep_ms(tms) usleep(tms*1000)
#define CLOCKTREE_CLKSRC_INTERNAL 0
static const int32_t UNITAPI_OK = 0;
static const int32_t FMC134_CLOCKTREE_ERR_OK = 0;
static const int32_t FMC134_ADC_ERR_OK       = 0;
static const int32_t FMC134_ERR_ADC_INIT     = 1;

int32_t Fmc134Cpld::default_clocktree_init()
{
  printf("*** default_clocktree_init ***\n");
  uint32_t dword;
  uint32_t dword2;
  uint32_t samplingrate_setting;
  unsigned i2c_unit;
  int32_t rc = UNITAPI_OK;     

  unsigned clockmode = CLOCKTREE_CLKSRC_INTERNAL;

  printf("Configured the sampling rate to 3200MSPs\n");
  samplingrate_setting = 0x6020000;                       //0x6020000 default N=/32
  rc = internal_ref_and_lmx_enable(i2c_unit, clockmode);

  usleep(100000);

  // LMX Programming for 3.2GHz
  spi_write(i2c_unit, LMX_SELECT,  5, 0x4087001);    // Force a Reset (default from codebuilder) 0x021F7001 << from data sheet default

  spi_write(i2c_unit, LMX_SELECT, 13, 0x4080C10);    // FOR 100MHz PDF  DLD TOL 1.7ns  0x4080C10
  //spi_write(i2c_unit, LMX_SELECT, 13, 0x4081C10);  // FOR 50MHz PDF  DLD TOL 6ns  0x4081C10

  spi_write(i2c_unit, LMX_SELECT, 10, 0x210050C);
  spi_write(i2c_unit, LMX_SELECT,  9, 0x03C7C03); 
  spi_write(i2c_unit, LMX_SELECT,  8, 0x207DDBF);
  spi_write(i2c_unit, LMX_SELECT,  7, 0x004E211)     ;       // Works R/2 output on mux

  spi_write(i2c_unit, LMX_SELECT,  6, 0x000004C);
  //                if(rc!=UNITAPI_OK)     return rc;                                               // lOOK AT CHANGE mode TO USE CE pin... vco_sel_mode = 1

  spi_write(i2c_unit, LMX_SELECT,  5, 0x0030808);    // 0x0030800 = 68MHz < OSC_FREQ < 128M     0x005080 = OSC_Freq > 512MHz         0x0010800 = OSC_FREQ =< 64MHz
  //spi_write(i2c_unit, LMX_SELECT,  5, 0x0030A80);  // = 1600MHz

  spi_write(i2c_unit, LMX_SELECT,  4, 0x0000000);

  spi_write(i2c_unit, LMX_SELECT,  3, 0x20040BE);   // 68B4: A=45 B=40  63B4: A=45 B=35   6DBC: A=47 B=45
  //spi_write(i2c_unit, LMX_SELECT,  3, 0x2002DB4);  //      = 1600MHz
  //              if(rc!=UNITAPI_OK)     return rc;                                               // OUT A and OUT B _PD deasserted  With the LR pullup a setting of 45:45 gives teh best noise/snr performance
  // original A = B = 0x2006DB4 = 45:45   0x20068A0 = 40:40   0x200638C = 35:35    0x2005E78 = 30:30    0x2004F3C = 15:15    0x2004003 = 0:0(nNonFunc)


  spi_write(i2c_unit, LMX_SELECT,  2, 0x0FD0902);    // 0,0,OSCx2 = 0, 0, CPP=1, 1, PLL denom dont care 
  if(rc!=UNITAPI_OK)     return rc;                                               //default is 0x0FD0902

  spi_write(i2c_unit, LMX_SELECT,  1, 0xF800001);    // C6000001  Rdivider = 1 no division   C6000008  Rdivider = 8 (800 MHz Ref 100PFD)    C6000009  Rdivider = 9 (900MHz Ref 100PFD)
  //if(rc!=UNITAPI_OK)     return rc;                                               // 0xC640000 <<< VCO_SEL 2
                                                                                                
  //0xC200001 is default

  // for 10MHz ref N=320 // R0 06140000  Register, Dither disabled,VCO-CAL_Enabled, 12Nin N 020 = 32, 010 
  //spi_write(i2c_unit, LMX_SELECT,  0, 0x6020000);  // R0 06020000  Register, Dither disabled,VCO-CAL_Enabled, 12Nin N 020 = 32, 010 
  spi_write(i2c_unit, LMX_SELECT,  0, samplingrate_setting); //      
  //if(rc!=UNITAPI_OK)     return rc;                                               // 0x6440000 <<< VCO_SEL 3
  // For 'R' = 1, Use ext red Directly 0x6280000-5MHz, 0x614000-10MHz 0x6020000-100Mhz 0x6040000-200Mhz

  usleep(300000);

  //spi_write(i2c_unit, LMX_SELECT,  0, 0x6020000);
  spi_write(i2c_unit, LMX_SELECT,  0, samplingrate_setting);

  // HMC Programming
  spi_write(i2c_unit, HMC_SELECT, 0x0, 0x1);         // Clear Reset

  spi_write(i2c_unit, HMC_SELECT, 0x1, 0x1 );                // Chip enable
        
  spi_write(i2c_unit, HMC_SELECT, 0x2, 0x91);                // Enable buffers 1, 5, and 8 x91 default
        
  spi_write(i2c_unit, HMC_SELECT, 0x3, 0x1A);                // Use internal DC bias string, no internal LVPECL term, 100 ohm differential input term, toggle RFBUF XOR
  if(rc!=UNITAPI_OK)     return rc;                                               //default 1A
        
  spi_write(i2c_unit, HMC_SELECT, 0x4, 0x00 );               // (x05) 3dBm gain FOR BRING-UP ONLY!!!
        
   spi_write(i2c_unit, HMC_SELECT, 0x5, 0x3A);                // "Biases" with reserved values...

 // LMK Programming
 rc = reset_clock_chip(i2c_unit);                                                // Reset clock chip

 usleep(5000);

 spi_write(i2c_unit, LMK_SELECT, 0x000 , 0x80 );    // Force a Reset
 spi_write(i2c_unit, LMK_SELECT, 0x000 , 0x00 );    // Clear reset
 spi_write(i2c_unit, LMK_SELECT, 0x000 , 0x10 );    // Force SPI to be 4-Wire
 spi_write(i2c_unit, LMK_SELECT, 0x148 , 0x33 );    // CLKIN_SEL0_MUX Configured as LMK MISO Push Pull Output
 spi_write(i2c_unit, LMK_SELECT, 0x002 , 0x00 );    // POWERDOWN Disabled (Normal Operation)     
 // CLK0/1 Settings GBTCLK0 and GBTCLK1 M2C LVDS both at 320MHz
 spi_write(i2c_unit, LMK_SELECT, 0x100 , 0x0A );    // DCLK0_1_DIV DIV_BY_10 = 320MHz 
 spi_write(i2c_unit, LMK_SELECT, 0x101 , 0x00 );    // DCLK0_1_DDLY = 0
 spi_write(i2c_unit, LMK_SELECT, 0x102 , 0x70 );    // 0 1 1 1  0 0 0 0             CLKout0_1 active, Hi-perf_out, Hi-Perf_In, Dig_Delay_Powered_down, DCLK0_1_DDLY[9:8] = 0, DCLK0_1_DIV[9:8 = 0
  spi_write(i2c_unit, LMK_SELECT, 0x103 , 0x40 );    // 0 1 0 0      0 0 0 0         n/a, halfstep delay PD, CLK0 = DCLK, DCLK0 active, Dclk use divider, no_duty_cyc_cor, DCLK0_Norm_Polarity, DCLK0_No_Halfstep
 spi_write(i2c_unit, LMK_SELECT, 0x104 , 0x00 );    // 0 0 0 0  0 0 0 0             n/a, n/a, CLK1 = DCLK, DCLK1 active, SCLK_DIS_MODE = 00, DCLK1_Norm_Polarity, DCLK1_No_Halfstep
 spi_write(i2c_unit, LMK_SELECT, 0x105 , 0x00 );    // 0 0 0 0  0 0 0 0             n/a, n/a, No_analog_Delay, 00000= analog delay
 spi_write(i2c_unit, LMK_SELECT, 0x106 , 0x00 );    // 0 0 0 0  0 0 0 0             n/a, n/a, n/a, n/a, 0000 = digital delay 
 spi_write(i2c_unit, LMK_SELECT, 0x107 , 0x11 );    // 0 0 0 1  0 0 0 1             LVDS, LVDS 
 // CLK2/3 Settings Output to FPGA 160MHz and SYSREF     may want to turn off DCLK
 spi_write(i2c_unit, LMK_SELECT, 0x108 , 0x14 );    // DCLK2_3_DIV DIV_BY_20 = 160MHz 
 spi_write(i2c_unit, LMK_SELECT, 0x109 , 0x00 );    // DCLK2_3_DDLY = 0
 spi_write(i2c_unit, LMK_SELECT, 0x10A , 0x70 );    // 0 1 1 1  0 0 0 0             CLKout2_3 active, Hi-perf_out, Hi-Perf_In, Dig_Delay_Powered_down, DCLK2_DIV8,9 = 0 DCLK3_DIV8,9 = 0
 spi_write(i2c_unit, LMK_SELECT, 0x10B , 0x40 );    // 0 1 0 0      0 0 0 0         n/a, halfstep_delay_PD, CLK2 = DCLK, DCLK2 active, Dclk use divider, no_duty_cyc_cor, DCLK2_Norm_Polarity, DCLK2_No_Halfstep
 spi_write(i2c_unit, LMK_SELECT, 0x10C , 0x20 );    // 0 0 1 0  0 0 0 0             n/a,  na, CLK3 = SCLK, DCLK3 active, SCLK_DIS_MODE = 00, DCLK3_Norm_Polarity, DCLK3_No_Halfstep
 spi_write(i2c_unit, LMK_SELECT, 0x10D , 0x00 );    // 0 0 0 0  0 0 0 0             n/a, n/a, SYSREF Analog_Delay disable, analog delay = 00000
 spi_write(i2c_unit, LMK_SELECT, 0x10E , 0x00 );    // 0 0 0 0  0 0 0 0             n/a, n/a, n/a, n/a, 0000 = digital delay 
 spi_write(i2c_unit, LMK_SELECT, 0x10F , 0x11 );    // 0 0 0 1  0 0 0 1             LVDS, LVDS
 
 // CLK4/5 Settings  CLK4 Power-down CLK5 = ADC1_SYSREF LVPECL
 spi_write(i2c_unit, LMK_SELECT, 0x110 , 0x20 );    // DCLK4_5_DIV DIV_BY_16 = 200MHz - not used
 spi_write(i2c_unit, LMK_SELECT, 0x111 , 0x00 );    // DCLK4_5_DDLY = 0
 spi_write(i2c_unit, LMK_SELECT, 0x112 , 0x70 );    // 0 1 1 1  0 0 0 0             CLKout4_5 active, Hi-perf_out, Hi-Perf_In, Dig_Delay_Powered_down, DCLK4_DIV8,9 = 0 DCLK5_DIV8,9 = 0
 spi_write(i2c_unit, LMK_SELECT, 0x113 , 0x40 );    // 0 1 0 1      0 0 0 0         n/a, halfstep_delay_PD, CLK4 = DCLK, DCLK4_5_PD, DCLK4_5_BYP, no_duty_cyc_cor, DCLK4_Norm_Polarity, DCLK4_No_Halfstep
 spi_write(i2c_unit, LMK_SELECT, 0x114 , 0x20 );    // 0 0 1 0  0 0 0 0             n/a,  na, CLK5 = SYSREF, SCLK4_5_PD active, SCLK_DIS_MODE = 00, DCLK3_Norm_Polarity, DCLK3_No_Halfstep
 spi_write(i2c_unit, LMK_SELECT, 0x115 , 0x00 );    // 0 0 0 0  0 0 0 0             n/a, n/a, No_analog_Delay, 00000= analog delay
 spi_write(i2c_unit, LMK_SELECT, 0x116 , 0x00 );    // 0 0 0 0  0 0 0 0             n/a, n/a, n/a, n/a, 0000 = digital delay 
 spi_write(i2c_unit, LMK_SELECT, 0x117 , 0x60 );    // 0 1 1 0  0 0 0 0             CLK5 = LVPECL 2000mV, clk4_OFF
 
 // CLK6/7 CLK6 = ************** POWERDOWN *********** ADC1 CLOCK @ 3200MHz
 spi_write(i2c_unit, LMK_SELECT, 0x118 , 0x02 );    // DCLK6_7_DIV DIV_BY_2 = 1600MHz  - not used
 spi_write(i2c_unit, LMK_SELECT, 0x119 , 0x00 );    // DCLK6_7_DDLY = 0
 spi_write(i2c_unit, LMK_SELECT, 0x11A , 0x70 );    // 0 1 1 1  0 0 0 0             CLKout6 active, Hi-perf_out, Hi-Perf_In, Dig_Delay_Powered_down, DCLK6_DIV 8,9 = 0 DCLK7_DIV 8,9 = 0
 spi_write(i2c_unit, LMK_SELECT, 0x11B , 0x48 );    // 0 1 0 0      1 0     0 0            n/a, halfstep delay PD, CLK6 = DCLK, DCLK6 active, Dclk6_BYPASS_DIV, no_duty_cyc_cor, DCLK6_Norm_Polarity, DCLK6_No_Halfstep
 spi_write(i2c_unit, LMK_SELECT, 0x11C , 0x30 );    // 0 0 1 1  0 0 0 0             n/a,  na, CLK7 = SYSCLK, DCLK7_PD, SCLK_DIS_MODE = 00, DCLK7_Norm_Polarity, DCLK7_No_Halfstep
 spi_write(i2c_unit, LMK_SELECT, 0x11D , 0x00 );    // 0 0 0 0  0 0 0 0             n/a, n/a, No_analog_Delay, 00000= analog delay
 spi_write(i2c_unit, LMK_SELECT, 0x11E , 0x00 );    // 0 0 0 0  0 0 0 0             n/a, n/a, n/a, n/a, 0000 = digital delay 
 spi_write(i2c_unit, LMK_SELECT, 0x11F , 0x00 );    // 0 0 0 0  0 0 0 0             Off
 
 // CLK8/9 CLK8 = ************** POWERDOWN *********** ADC0 CLOCK @ 3200MHz
 spi_write(i2c_unit, LMK_SELECT, 0x120 , 0x02 );    // DCLK8_9_DIV DIV_BY_2 = 1600MHz  - not used
 spi_write(i2c_unit, LMK_SELECT, 0x121 , 0x00 );    // DCLK8_9_DDLY = 0
 spi_write(i2c_unit, LMK_SELECT, 0x122 , 0x70 );    // 0 1 1 1  0 0 0 0             CLKout8 active, Hi-perf_out, Hi-Perf_In, Dig_Delay_Powered_down, DCLK8_DIV 8,9 = 0 DCLK9_DIV 8,9 = 0
 spi_write(i2c_unit, LMK_SELECT, 0x123 , 0x48 );    // 0 1 0 0      1 0     0 0            n/a, halfstep delay PD, CLK8 = DCLK, DCLK8 active, Dclk8_BYPASS_DIV, no_duty_cyc_cor, DCLK8_Norm_Polarity, DCLK8_No_Halfstep
 spi_write(i2c_unit, LMK_SELECT, 0x124 , 0x30 );    // 0 0 1 1  0 0 0 0             n/a,  na, CLK9 = SYSCLK, DCLK9_PD, SCLK_DIS_MODE = 00, DCLK9_Norm_Polarity, DCLK9_No_Halfstep
 spi_write(i2c_unit, LMK_SELECT, 0x125 , 0x00 );    // 0 0 0 0  0 0 0 0             n/a, n/a, No_analog_Delay, 00000= analog delay
 spi_write(i2c_unit, LMK_SELECT, 0x126 , 0x00 );    // 0 0 0 0  0 0 0 0             n/a, n/a, n/a, n/a, 0000 = digital delay 
 spi_write(i2c_unit, LMK_SELECT, 0x127 , 0x00 );    // 0 0 0 0  0 0 0 0             Off
 
 // CLK10/11 Settings  CLK10 Power-down CLK11 = ADC0_SYSREF LVPECL
 spi_write(i2c_unit, LMK_SELECT, 0x128 , 0x20 );    // DCLK10_11_DIV DIV_BY_16 = 200MHz - not used
 spi_write(i2c_unit, LMK_SELECT, 0x129 , 0x00 );    // DCLK10_11_DDLY = 0
 spi_write(i2c_unit, LMK_SELECT, 0x12A , 0x70 );    // 0 1 1 1  0 0 0 0             CLKout10_11 active, Hi-perf_out, Hi-Perf_In, Dig_Delay_Powered_down, DCLK10_11_DIV8,9 = 0 DCLK10_11_DIV8,9 = 0
 spi_write(i2c_unit, LMK_SELECT, 0x12B , 0x40 );    // 0 1 0 1      0 0 0 0         n/a, halfstep_delay_PD, CLK10 = DCLK, DCLK10_11_PD, DCLK10_11_BYP, no_duty_cyc_cor, DCLK10_Norm_Polarity, DCLK10_No_Halfstep
 spi_write(i2c_unit, LMK_SELECT, 0x12C , 0x20 );    // 0 0 1 0  0 0 0 0             n/a,  na, CLK11 = SYSREF, SCLK10_11_PD active, SCLK_DIS_MODE = 00, DCLK11_Norm_Polarity, DCLK11_No_Halfstep
 spi_write(i2c_unit, LMK_SELECT, 0x12D , 0x00 );    // 0 0 0 0  0 0 0 0             n/a, n/a, No_analog_Delay, 00000= analog delay
 spi_write(i2c_unit, LMK_SELECT, 0x12E , 0x00 );    // 0 0 0 0  0 0 0 0             n/a, n/a, n/a, n/a, 0000 = digital delay 
 spi_write(i2c_unit, LMK_SELECT, 0x12F , 0x60 );    // 0 1 0 1  0 0 0 0             CLK11 = LVPECL-2000mV, clk10_OFF  << This may Change to lower Amplitude: 0x40LVPECV-1600, 0x50=LVPECL-2000 0x60=LCPECL 
 
 // CLK12/13 GBTCLK2  and GBTCLK3 M2C LVDS both at 320MHz
 spi_write(i2c_unit, LMK_SELECT, 0x130 , 0x0A );    // DIV_CLKOUT0 DIV_BY_10 = 320MHz 
 spi_write(i2c_unit, LMK_SELECT, 0x131 , 0x00 );    // delay unused
 spi_write(i2c_unit, LMK_SELECT, 0x132 , 0x70 );    // 0 1 1 1  0 0 0 0             CLKout0 active, Hi-perf_out, Hi-Perf_In, Dig_Delay_Powered_down, DCLK0_DIV8, 9 = 0 DCLK1_DIV8, 9 = 0
 spi_write(i2c_unit, LMK_SELECT, 0x133 , 0x40 );    // 0 1 0 0      0 0 0 0         n/a, halfstep delay PD, CLK0 = DCLK, DCLK0 active, Dclk use divider, no_duty_cyc_cor, DCLK0_Norm_Polarity, DCLK0_No_Halfstep
 spi_write(i2c_unit, LMK_SELECT, 0x134 , 0x00 );    // 0 0 0 0  0 0 0 0             n/a, na, CLK1 = DCLK,  DCLK1 active, SCLK_DIS_MODE = 00, DCLK1_Norm_Polarity, DCLK1_No_Halfstep
 spi_write(i2c_unit, LMK_SELECT, 0x135 , 0x00 );    // 0 0 0 0  0 0 0 0             n/a, n/a, No_analog_Delay, 00000= analog delay
 spi_write(i2c_unit, LMK_SELECT, 0x136 , 0x00 );    // 0 0 0 0  0 0 0 0             n/a, n/a, n/a, n/a, 0000 = digital delay 
 spi_write(i2c_unit, LMK_SELECT, 0x137 , 0x11 );    // 0 0 0 1  0 0 0 1             LVDS, LVDS
 
 // the default mode uses the LMX2581 as a clock source so PLL1 must be disabled

 // Select VCO1 PLL1 source
 spi_write(i2c_unit, LMK_SELECT, 0x138 , 0x40 );    // 0 1 0 0  0 0 0 0   CLKin1(externla VCO) Buf_osc_in, PowerDown 
 spi_write(i2c_unit, LMK_SELECT, 0x139 , 0x03 );    // SYSREF_MUX, SYSREF_Free_Running_Output     SYSREF MUST BE initially ON (TBD) 
 
 // SYSREF Divider
 spi_write(i2c_unit, LMK_SELECT, 0x13A , 0x01 );    // SYSREF_DIV(MS) SYSREF Divider    3200 / 320 = 10MHz
   spi_write(i2c_unit, LMK_SELECT, 0x13B , 0x40 );    // SYSREF_DIV(LS) SYSREF Divider
   
   // SYSREF Digital Delay
   spi_write(i2c_unit, LMK_SELECT, 0x13C , 0x00 );    // SYSREF_DDLY(MS) SYSREF Digital Delay  - Not Used
   spi_write(i2c_unit, LMK_SELECT, 0x13D , 0x08 );    // SYSREF_DDLY(LS) SYSREF Digital Delay  - Not Used     
   
   spi_write(i2c_unit, LMK_SELECT, 0x13E , 0x00 );    // SYSREF_PULSE_CNT 8 Pulses - Not Used
   
   // PLL2
   spi_write(i2c_unit, LMK_SELECT, 0x13F , 0x00 );    // (defaults not used) FB_CTRL PLL2_FB=prescaler, PLL1_FB=OSCIN   This is default for internal Oscillator, this changes on EXT osc   
   spi_write(i2c_unit, LMK_SELECT, 0x140 , 0xF1 );    // 1 1 1 1   0 0 0 0    PLL1_PD, VCO_LDO_PD, VCO_PD, OSCin_PD, All SYSREF Normal
   //if(rc!=UNITAPI_OK)     return rc;                                               // 0x01 default
   //try 0xf1
   spi_write(i2c_unit, LMK_SELECT, 0x141 , 0x00 );    // Dynamic digital delay step = no adjust
   spi_write(i2c_unit, LMK_SELECT, 0x142 , 0x00 );    // DIG_DLY_STEP_CNT No Adjustment of Digital Delay     
   
   spi_write(i2c_unit, LMK_SELECT, 0x143 , 0x70 );    // SYNC_SYSREF SYNC functionality enabled, prevent SYNC pin and DLD flags from generating SYNC event
   //if(rc!=UNITAPI_OK)     return rc;                                               // DCLK12, DCLK10, DCLK8 do not re-sync during a sync event   ((*** SAME as 120 ***))??
   spi_write(i2c_unit, LMK_SELECT, 0x144 , 0xFF );    // DISABLE_DCLK_SYNC Prevent SYSREF clocks from synchronizing     

   // new R counter sync function
   spi_write(i2c_unit, LMK_SELECT, 0x145 , 0x00 );    // No Information yet and probably not applicable     
   
   spi_write(i2c_unit, LMK_SELECT, 0x146 , 0x00 );    // CLKIN_SRC No AutoSwitching of clock inputs, all 3 CLKINx pins are set t0 Bipolar
   
   // fmc134 LMK Clock inputs are : clkin0 = trigger, clkin1 = LMX_OUT, clkin2 = off, OSCin = low frequency LMK, OSCout not used
   // Buffer LMX PLL as Clock Source
   if (clockmode == CLOCKTREE_CLKSRC_INTERNAL)                     
     {                                                                                       
       spi_write(i2c_unit, LMK_SELECT, 0x147 , 0x10 );  // 0 0 0 1  1 1 1 1  CLK_SEL_POL=hi, CLKIN_MUX_SEL= CLKIN_1 Manual = LMX2581E_ !INVERT, CLKIN1=Fin CLKIN0=SYSREF MUX
       printf("Using LMX2581 PLL as Clock Source\n") ;
       // !!!JOHN!!! explicitly clear ext_sample_clk_3p3 in CPLD !!!JOHN!!!
     }

   spi_write(i2c_unit, LMK_SELECT, 0x148 , 0x33 );    // CLKIN_SEL0_MUX Configured as LMK MISO Push Pull Output
   spi_write(i2c_unit, LMK_SELECT, 0x149 , 0x00 );    // LKIN_SEL1=input     << Not used
   spi_write(i2c_unit, LMK_SELECT, 0x14A , 0x00 );    //  RESET_MUX RESET Pin=Input Active High
   spi_write(i2c_unit, LMK_SELECT, 0x14B , 0x02 );    // default      Disabled holdover DAC but leave at 0x200
   spi_write(i2c_unit, LMK_SELECT, 0x14C , 0x00 );    // default      disabled but leave DAC at midscale 0x0200
   spi_write(i2c_unit, LMK_SELECT, 0x14D , 0x00 );    // default      DAC_TRIP_LOW Min Voltage to force HOLDOVER
   spi_write(i2c_unit, LMK_SELECT, 0x14E , 0x00 );    // default      DAC_TRIP_HIGH Mult=4 Max Voltage to force HOLDOVER
   spi_write(i2c_unit, LMK_SELECT, 0x14F , 0x7F );    // default      DAC_UPDATE_CNTR
   spi_write(i2c_unit, LMK_SELECT, 0x150 , 0x00 );    // default      HOLDOVER_SET HOLDOVER disable  << NEW Functionality
   spi_write(i2c_unit, LMK_SELECT, 0x151 , 0x02 );    // default      HOLD_EXIT_COUNT(MS)
   spi_write(i2c_unit, LMK_SELECT, 0x152 , 0x00 );    // default      HOLD_EXIT_COUNT(LS)
   
   //PLL1 CLKIN0 R Divider Not Used
   spi_write(i2c_unit, LMK_SELECT, 0x153 , 0x00 );    //not used      CLKIN0_DIV (MS)
   spi_write(i2c_unit, LMK_SELECT, 0x154 , 0x80 );    //not used      CLKIN0_DIV (LS)
   
   //PLL1 CLKIN1 R Divider Not Used 
   spi_write(i2c_unit, LMK_SELECT, 0x155 , 0x00 );    // Not Used     CLKIN1_DIV (MS)     
   spi_write(i2c_unit, LMK_SELECT, 0x156 , 0X80 );    // Not Used     CLKIN1_DIV (LS)   

   //PLL1 CLKIN2 R Divider not Used 
   spi_write(i2c_unit, LMK_SELECT, 0x157 , 0x03 );    // Not Used     CLKIN2_DIV (MS) 
        
   spi_write(i2c_unit, LMK_SELECT, 0x158 , 0xE8 );    // Not Used     CLKIN2_DIV (LS)      
   
   // This is part of a secondary configuration
   // LMX2581 Low frequency Output for use with LMK04832 VCO & PLL2, nominal frequency 500MHz 
   // configured for 100MHz reference to PLL2
   // PLL1 N divider, Divide 500MHz VCSO down to PDF
   spi_write(i2c_unit, LMK_SELECT, 0x159 , 0x00 );    // PLL1_NDIV (MS)  PLL1 Ndivider = 5000 for 100HHz PDF    
   spi_write(i2c_unit, LMK_SELECT, 0x15A , 0x05 );    // PLL1_NDIV (LS)       500MHz/5 = 100MHz PFD
   
   // PLL1 Configuration
   spi_write(i2c_unit, LMK_SELECT, 0x15B , 0xF4 );    // PLL1 Pasive CPout1 tristate, Pos Slope, 50uA 
   spi_write(i2c_unit, LMK_SELECT, 0x15C , 0x20 );    // Default not used 
   spi_write(i2c_unit, LMK_SELECT, 0x15D , 0x00 );    // Default not used 
   spi_write(i2c_unit, LMK_SELECT, 0x15E , 0x00 );    // default not used 
   spi_write(i2c_unit, LMK_SELECT, 0x15F , 0x03 );    // Pasive Forced Logic Low Push_Pull 
   
   //      In the default usage PLL2 is pasivated
   // default mode is LMK provides a 400MHz reference clock and the PLL multiples it up to 3200? TBD 
   // PLL2 onfigured to lock VCO1 at 3000MHz to 500MHz from LMX with a PFD of 125MHz, (4N * 6P = 24) * 125MHz = 3000MHz
   // a prescale value of 6 allows the PLL2 N and R to match 
   spi_write(i2c_unit, LMK_SELECT, 0x160 , 0x00 ) ;   // PLL2_RDIV (MS) PLL2 Reference Divider = 4 refference frequency = 125MHz   
   spi_write(i2c_unit, LMK_SELECT, 0x161 , 0x04 ) ;   // PLL2_RDIV (LS)     
   spi_write(i2c_unit, LMK_SELECT, 0x162 , 0xCC ) ;   // D0 changed to 0xCC per new Migration doc
   spi_write(i2c_unit, LMK_SELECT, 0x163 , 0x00 ) ;   // PLL2_NCAL (HI) Only used during CAL
   spi_write(i2c_unit, LMK_SELECT, 0x164 , 0x00 ) ;   // PLL2_NCAL (MID)              
   spi_write(i2c_unit, LMK_SELECT, 0x165 , 0x04 ) ;   // PLL2_NCAL (LOW)
   
   // the following 5 writes are out of sequence per the TI programming sequence recomendations in the data sheet
   spi_write(i2c_unit, LMK_SELECT, 0x145 , 0x00 ) ;   // << Ignore, modify R divider Sync is needed
   spi_write(i2c_unit, LMK_SELECT, 0x171 , 0xAA ) ;   //      << Specified by TI
   spi_write(i2c_unit, LMK_SELECT, 0x171 , 0x02 ) ;   //      << Specified by TI
   
   spi_write(i2c_unit, LMK_SELECT, 0x17C , 0x15 ) ;   // OPT_REG1     **** VERIFY when new data sheet arives
   spi_write(i2c_unit, LMK_SELECT, 0x17D , 0x33 ) ;   // OPT_REG2     **** VERIFY when new data sheet arives
   
   spi_write(i2c_unit, LMK_SELECT, 0x166 , 0x00 ) ;   // PLL2_NDIV (HI) Allow CAL     
   spi_write(i2c_unit, LMK_SELECT, 0x167 , 0x00 ) ;   // PLL2_NDIV (MID) PLL2 N-Divider     
   spi_write(i2c_unit, LMK_SELECT, 0x168 , 0x04 ) ;   //      // PLL2_NDIV (LOW) Cal after writing this register     >>P = 3, N = 8  (24 * 125Mhz_ref = 3G)   
   spi_write(i2c_unit, LMK_SELECT, 0x169 , 0x49 ) ;   // PLL2_SETUP Window 3.7nS,  I(cp)=1.6mA, Pos Slope, CP ! Tristate, Bit 0 always 1  
   // 1.6mA gives better close in phase  noise than 3.2mA

   spi_write(i2c_unit, LMK_SELECT, 0x16A , 0x00 ) ;   // PLL2_LOCK_CNT (MS)      
   spi_write(i2c_unit, LMK_SELECT, 0x16B , 0x20 ) ;   // PLL2_LOCK_CNT (LS)  PD must be in lock for 16 cycles    
   spi_write(i2c_unit, LMK_SELECT, 0x16C , 0x00 ) ;   // PLL2_LOOP_FILTER_R Disable Internal Resistors        << Uses externla Loop Filter 
   // R3 = 200 Ohms  R4 = 200 Ohms

   spi_write(i2c_unit, LMK_SELECT, 0x16D , 0x00 ) ;   // PLL2_LOOP_FILTER_C Disable Internal Caps             << uses externla loop filter
   // C3 = 10pF  C4 = 10pF

   spi_write(i2c_unit, LMK_SELECT, 0x16E , 0x12 ) ;   // // STATUS_LD2_MUX LD2=Locked   Push Pull Output
   
   // this disables PLL2
   if(1) {
     spi_write(i2c_unit, LMK_SELECT, 0x173 , 0x60 ) ;   // 0 1 1 0 0 0 0 0  0x60 PLL2_Prescale_PD PLL2_PD 
     //if(rc!=UNITAPI_OK)     return rc;                                               //0x00 original
     printf("LMK PLL2 Powered Down\n") ;
   }

   // This Enables PLL2
   if(0) {
     spi_write(i2c_unit, LMK_SELECT, 0x173 , 0x00 ) ;   // PLL2_MISC PLL2 Active, normal opperation  
     printf("LMK PLL2 Active \n") ;
   }       

   usleep(100000);  // allow PLL to lock, not required in buffermode but does not hurt

   // Clear LMK PLL2 Erros regardless of if we use them
   spi_write(i2c_unit, LMK_SELECT, 0x183, 0x01 ) ;
   if(rc!=UNITAPI_OK)     return rc;
   spi_write(i2c_unit, LMK_SELECT, 0x183, 0x00 ) ;
   if(rc!=UNITAPI_OK)     return rc;

   // IF we are using LMK04832 PLL2 then wait500ms  to see if we ever go out of lock
   if (clockmode == CLOCKTREE_CLKSRC_INTERNAL )
     {
       unitapi_sleep_ms(500);          //      Look for half a sec to see if PLL is unlocked

       // verify LMK04832 PLL2 status
       spi_read(i2c_unit, LMK_SELECT, 0x183, &dword ) ;
       if(rc!=UNITAPI_OK)
         return rc;

       dword2 = (dword & 0x07);
       if ((dword&0x02)!=0x02) {
         //printf("LMK04832 PLL2 not locked!!! reg 0x183 = 0x%X\n",dword2);      
         // not an error
         //return FMC134_CLOCKTREE_ERR_CLK0_PLL_NOT_LOCKED;
       }
       else {
         printf("PLL2 locked!!! \n");
       }
     }
   // try to sync all the output dividers
   // SYNC_MODE enable to SYNC event
   // SYSREF_CLR = 1
   // SYNC_1SHOT_EN = 1
   // SYNC_POL = 0 (Normal)
   // SYNC_EN = 1
   // SYNC_MODE = 1 (sync_event_generatedfrom SYNC pin)
   spi_write(i2c_unit, LMK_SELECT, 0x143, 0xD1);

   // change SYSREF_MUX to normal SYNC (0)
   spi_write(i2c_unit, LMK_SELECT, 0x139, 0x00);

   // Enable dividers reset
   spi_write(i2c_unit, LMK_SELECT, 0x144, 0x00);

   //toggle the polarity (keep SYSREF_CLR active)
   spi_write(i2c_unit, LMK_SELECT, 0x143, 0xF1);

   unitapi_sleep_ms(10);

   spi_write(i2c_unit, LMK_SELECT, 0x143, 0xD1);
   // disable dividers
   spi_write(i2c_unit, LMK_SELECT, 0x144, 0xFF);

   // change SYSREF_MUX back to continuous
   spi_write(i2c_unit, LMK_SELECT, 0x139, 0x03);

   // restore SYNC_MODE & remove SYSREF_CLR
   spi_write(i2c_unit, LMK_SELECT, 0x143, 0x50);

   printf("*** default_clocktree_init done ***\n");
   return 0;
}

int32_t Fmc134Cpld::internal_ref_and_lmx_enable(uint32_t i2c_unit, uint32_t clockmode)
{
        uint32_t dword = 0;
        int32_t rc = UNITAPI_OK;
        //        int32_t oscmode = 1;

        printf("\n\n Setting Clock_Mode \n");

        // Set one byte per cycle
        // rc = unitapi_write_register(i2c_unit, I2C_BAR_CTRL+0x05, 0x00);
        // if(rc!=UNITAPI_OK)
        //         return rc;

        // Read, Modify, Write to avoid clobbering any other register settings
        i2c_read(i2c_unit, cpld_address + 2, &dword);
        if(rc!=UNITAPI_OK)
                return rc;

        switch(clockmode) {
        case 0:                         // Internal Reference
                printf("Internal Reference Mode \n");
                // Set bits 3, 1, and 0
                dword |= 0xB;
                i2c_write(i2c_unit, cpld_address + 2, dword);
                if(rc!=UNITAPI_OK)
                        return rc;

                // Read, Modify, Write to avoid clobbering any other register settings
                i2c_read(i2c_unit, cpld_address + 1, &dword);
                if(rc!=UNITAPI_OK)
                        return rc;

                // configure the switch for the internal clock (CPLD address 1 bit 2)
                dword |= 1<<2;
                i2c_write(i2c_unit, cpld_address + 1, dword);
                if(rc!=UNITAPI_OK)
                        return rc;
                break;

        case 1:
                printf("External Sample Clock \n");
                // turn off 0sc, ref switch and LMX enable0
                dword = 0xC4;
                i2c_write(i2c_unit, cpld_address + 2, dword);
                if(rc!=UNITAPI_OK)
                        return rc;

                // Read, Modify, Write to avoid clobbering any other register settings
                i2c_read(i2c_unit, cpld_address + 1, &dword);
                if(rc!=UNITAPI_OK)
                        return rc;

                // Clear bit 2
                dword &= 0xFB; 
                i2c_write(i2c_unit, cpld_address + 1, dword);
                if(rc!=UNITAPI_OK)
                        return rc;
                break;

        case 2:         
                printf("External Reference Mode \n");
                // turn off 0sc, point at ext ref enable LMX bits 3, 1, and 0
                dword = 0xCC;
                i2c_write(i2c_unit, cpld_address + 2, dword);
                if(rc!=UNITAPI_OK)
                        return rc;

                // Read, Modify, Write to avoid clobbering any other register settings
                i2c_read(i2c_unit, cpld_address + 1, &dword);
                if(rc!=UNITAPI_OK)
                        return rc;

                // set bit 2
                dword |= 0x04;
                i2c_write(i2c_unit, cpld_address + 1, dword);
                if(rc!=UNITAPI_OK)
                        return rc;
                break;

        case 3:         
                printf("Stacked Internal Reference Sourcing Mode \n");
                // turn ON 0sc, point at ext ref enable LMX bits 3, 1, and 0
                dword = 0xCD;
                i2c_write(i2c_unit, cpld_address + 2, dword);
                if(rc!=UNITAPI_OK)
                        return rc;

                // Read, Modify, Write to avoid clobbering any other register settings
                i2c_read(i2c_unit, cpld_address + 1, &dword);
                if(rc!=UNITAPI_OK)
                        return rc;

                // set bit 2    (Internal Sample Clock)
                dword |= 0x04;
                i2c_write(i2c_unit, cpld_address + 1, dword);
                if(rc!=UNITAPI_OK)
                        return rc;
                break;

        default:
                // Clear bits 3, 1, and 0
                dword &= 0xF4;
                i2c_write(i2c_unit, cpld_address + 2, dword);
                if(rc!=UNITAPI_OK)
                        return rc;
                break;
        }

        return FMC134_CLOCKTREE_ERR_OK;
}

int32_t Fmc134Cpld::reset_clock_chip(int32_t)
{
        uint32_t dword;
        int32_t rc = UNITAPI_OK;;

        // Set one byte per cycle
        // rc = unitapi_write_register(i2c_unit, I2C_BAR_CTRL+0x05, 0x00);
        // if(rc!=UNITAPI_OK)
        //         return rc;

        // Read, Modify, Write to avoid clobbering any other register settings
        i2c_read(i2c_unit, cpld_address + 1, &dword);
        if(rc!=UNITAPI_OK)
                return rc;

        // Set reset bit
        dword |= 0x08;
        i2c_write(i2c_unit, cpld_address + 1, dword);
        if(rc!=UNITAPI_OK)
                return rc;

        // Clear reset bit
        dword &= 0xF7;
        i2c_write(i2c_unit, cpld_address + 1, dword);
        if(rc!=UNITAPI_OK)
                return rc;

        return 0;
}

int32_t Fmc134Cpld::default_adc_init()
{
  printf("*** default_adc_init ***\n");
  uint32_t sampleMode=1;
  uint32_t adc_txemphasis=0;

        uint32_t i2c_unit;
        uint32_t dword0;
        int32_t rc = UNITAPI_OK;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // ADC Initializaton
        printf("\nClearing ADC RESET pins\n");

        // Set one byte per cycle
        // rc = unitapi_write_register(i2c_unit, I2C_BAR_CTRL+0x05, 0x00);
        // if(rc!=UNITAPI_OK)
        //         return rc;

        // Read, modify, write to avoid clobbering any other register settings
        i2c_read(i2c_unit, cpld_address + 1, &dword0);
        if(rc!=UNITAPI_OK)
                return rc;

        // Force a clear on the ADC0 Reset Pin
        dword0 &= 0xFC;
        i2c_write(i2c_unit, cpld_address + 1, dword0);
        if(rc!=UNITAPI_OK) return rc;


        unitapi_sleep_ms(2);            

        // Reset part
        spi_write(i2c_unit, ADC_SELECT_BOTH, 0x0000, 0xB0);
        if(rc!=UNITAPI_OK) return rc;

        // Set the D Clock  and SYSREF input pins to LVPECL
        spi_write(i2c_unit, ADC_SELECT_BOTH, 0x002A, 0x06);
        if(rc!=UNITAPI_OK) return rc;

        // Set Timestamp input pins to LVPECL but do not enable timestamp
        spi_write(i2c_unit, ADC_SELECT_BOTH, 0x003B, 0x02);
        if(rc!=UNITAPI_OK) return rc;

        // Invert ADC0 Clock            (write to only ADC0)
        spi_write(i2c_unit, ADC0_SELECT, 0x02B7, 0x01);
        if(rc!=UNITAPI_OK) return rc;

        // Enable SYSREF Processor
        spi_write(i2c_unit, ADC_SELECT_BOTH, 0x0029, 0x20);
        if(rc!=UNITAPI_OK) return rc;
        spi_write(i2c_unit, ADC_SELECT_BOTH, 0x0029, 0x60);
        if(rc!=UNITAPI_OK) return rc;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // JESD Initializaton
        // Reset JESD during configuration
        spi_write(i2c_unit, ADC_SELECT_BOTH, 0x0200, 0x00);
        if(rc!=UNITAPI_OK) return rc;

        // Clear Cal Enable AFTER clearing JESD Enable during configuration
        spi_write(i2c_unit, ADC_SELECT_BOTH, 0x0061, 0x00);
        if(rc!=UNITAPI_OK) return rc;

        // Enable SYSREF Calibration while background calibration is disabled
        // Set 256 averages with 256 cycles per accumulation
        spi_write(i2c_unit, ADC_SELECT_BOTH, 0x02B1, 0x0F);
        if(rc!=UNITAPI_OK) return rc;

        // Start SYSREF Calibration
        spi_write(i2c_unit, ADC_SELECT_BOTH, 0x02B0, 0x01);
        if(rc!=UNITAPI_OK) return rc;

        unitapi_sleep_ms(500);

        // Read SYSREF Calibration status
        spi_read(i2c_unit, ADC0_SELECT, 0x02B4, &dword0);
        if(rc!=UNITAPI_OK) return rc;

        if ((dword0 & 0x2) == 0x2)
                printf("ADC0 SYSREF Calibration Done\n");
        else {
                printf("ADC0 SYSREF Calibration NOT Done!\n");
                return FMC134_ERR_ADC_INIT;
        }
        spi_read(i2c_unit, ADC1_SELECT, 0x02B4, &dword0);
        if(rc!=UNITAPI_OK) return rc;

        if ((dword0 & 0x2) == 0x2)
                printf("ADC1 SYSREF Calibration Done\n");
        else {
                printf("ADC1 SYSREF Calibration NOT Done!\n");
                return FMC134_ERR_ADC_INIT;
        }

        // Set CAL_BG to enable background calibration
        spi_write(i2c_unit, ADC_SELECT_BOTH, 0x0062, 0x02);                //**** BACKGROUND CALIBRATION ******
        if(rc!=UNITAPI_OK) return rc;                                   // 0x02 Background cal enabled

        // Set JMODE = 2 (or 0 for single channel mode)
        if (sampleMode == 0) {
                spi_write(i2c_unit, ADC_SELECT_BOTH, 0x0201, 0x02);
                if(rc!=UNITAPI_OK) return rc;
        } else if (sampleMode == 1) {
                spi_write(i2c_unit, ADC_SELECT_BOTH, 0x0201, 0x00);
                if(rc!=UNITAPI_OK) return rc;
        } else {
                printf("Unsupported Sample Mode!\n");
                return FMC134_ERR_ADC_INIT;
        }

        // Set K = 16
        spi_write(i2c_unit, ADC_SELECT_BOTH, 0x0202, 0x0F);
        if(rc!=UNITAPI_OK) return rc;

        // Keep output format as 2's complement and ENABLE Scrambler
        //spi_write(i2c_unit, ADC_SELECT_BOTH, 0x0204, 0x03);
        // Use binary offset output format and ENABLE Scrambler
        spi_write(i2c_unit, ADC_SELECT_BOTH, 0x0204, 0x01);
        if(rc!=UNITAPI_OK) return rc;

        // Set Cal Enable BEFORE setting JESD Enable after configuration
        spi_write(i2c_unit, ADC_SELECT_BOTH, 0x0061, 0x01);

        // Take JESD out of reset after configuration
        spi_write(i2c_unit, ADC_SELECT_BOTH, 0x0200, 0x01);
        if(rc!=UNITAPI_OK) return rc;

#if 1
        // full scale range ** this setting directly affects the ADC SNR  **
        spi_write(i2c_unit, ADC_SELECT_BOTH, 0x0030, 0xFF);        // NOTE this setting directly affects the ADC SNR
        if(rc!=UNITAPI_OK) return rc;                                                           // 0x0000 ~500mVp-p puts the max SNR at ~ 48.8dBFS
        spi_write(i2c_unit, ADC_SELECT_BOTH, 0x0031, 0xFF);        // 0xA4C4 ~725mVp-p puts the max SNR at ~ 55.5dBFS (Default value at reset)
        if(rc!=UNITAPI_OK) return rc;                                                           // 0xFFFF ~950mVp-p puts the max SNR at ~ 56.5dBFS
        spi_write(i2c_unit, ADC_SELECT_BOTH, 0x0032, 0xFF);
        if(rc!=UNITAPI_OK) return rc;
        spi_write(i2c_unit, ADC_SELECT_BOTH, 0x0033, 0xFF);
        if(rc!=UNITAPI_OK) return rc;
#endif

        // verify ADC1 is present, This verifys the SPI connection to ADC 1 is present
        unsigned dw[4];

        for(unsigned i=0; i<4; i++)
          spi_read(i2c_unit, ADC0_SELECT, 0x0030+i, &dw[i]) ;
        printf("Read FS_RANGE_0: %x %x %x %x\n",
               dw[0], dw[1], dw[2], dw[3]);

        for(unsigned i=0; i<4; i++)
          spi_read(i2c_unit, ADC1_SELECT, 0x0030+i, &dw[i]) ;
        printf("Read FS_RANGE_1: %x %x %x %x\n",
               dw[0], dw[1], dw[2], dw[3]);

#if 0
        // full scale range ** this setting directly affects the ADC SNR  **
        spi_write(i2c_unit, ADC_SELECT_BOTH, 0x0030, 0xff);        // NOTE this setting directly affects the ADC SNR
        if(rc!=UNITAPI_OK) return rc;                                                           // 0x0000 ~500mVp-p puts the max SNR at ~ 48.8dBFS
        spi_write(i2c_unit, ADC_SELECT_BOTH, 0x0031, 0xff);        // 0xA4C4 ~725mVp-p puts the max SNR at ~ 55.5dBFS (Default value at reset)
        if(rc!=UNITAPI_OK) return rc;                                                           // 0xFFFF ~950mVp-p puts the max SNR at ~ 56.5dBFS
        spi_write(i2c_unit, ADC_SELECT_BOTH, 0x0032, 0xff);
        if(rc!=UNITAPI_OK) return rc;
        spi_write(i2c_unit, ADC_SELECT_BOTH, 0x0033, 0xff);
        if(rc!=UNITAPI_OK) return rc;
#endif

        unitapi_sleep_ms(5);

        // Configure the transceiver pre-emphasis setting (0 to 0xF)
        spi_write(i2c_unit, ADC_SELECT_BOTH, 0x0048, adc_txemphasis);
        if(rc!=UNITAPI_OK) return rc;

        // Disable SYSREF Processor in ADC before turning off SYSREF outputs
        spi_write(i2c_unit, ADC_SELECT_BOTH, 0x0029, 0x00);
        if(rc!=UNITAPI_OK) return rc;

        spi_write(i2c_unit, LMK_SELECT, 0x12F , 0x00 );    // Disable Sysref to ADC 0
        if(rc!=UNITAPI_OK) return rc;
        spi_write(i2c_unit, LMK_SELECT, 0x117 , 0x00 );    // Disable Sysref to ADC 1
        if(rc!=UNITAPI_OK) return rc;

        printf("*** default_adc_init done ***\n");

        return FMC134_ADC_ERR_OK;
}

int32_t Fmc134Cpld::config_prbs(unsigned v)
{
  spi_write(i2c_unit, ADC_SELECT_BOTH, 0x0205, v);
  return FMC134_ADC_ERR_OK;
}

void Fmc134Cpld::adc_range(unsigned chip,unsigned fsrng)
{
  DevSel dev = (chip==0) ? ADC0 : ADC1;

  // const float scale = float(0xe000)/0.5;
  // int fsrng = int((fsvpp - 0.5) * scale + 0x2000);
  if (fsrng < 0x2000) fsrng = 0x2000;
  if (fsrng > 0xffff) fsrng = 0xffff;

  //  printf("Setting FS_RANGE %f Vpp [%x]\n", fsvpp, fsrng);

  spi_write(0, dev, 0x30, (fsrng&0xff));
  spi_write(0, dev, 0x31, (fsrng>>8));
  spi_write(0, dev, 0x32, (fsrng&0xff));
  spi_write(0, dev, 0x33, (fsrng>>8));

  unsigned dw[4];
  for(unsigned i=0; i<4; i++)
    spi_read(0, dev, 0x0030+i, &dw[i]) ;
  printf("Read FS_RANGE_0: %x %x %x %x\n",
    dw[0], dw[1], dw[2], dw[3]);
}
