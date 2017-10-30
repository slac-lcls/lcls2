#include "psdaq/hsd/FmcSpi.hh"
#include "psdaq/hsd/Globals.hh"

#include <unistd.h>
#include <stdio.h>

using namespace Pds::HSD;

//#define DBUG

#define FMC12X_ADC_TM_FLASH11       0x1                                                      /*!< ADC Test Mode, Flashing 1 every ten 0 patterns */

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

#define _WAIT usleep(10000)

enum SPIDevice { AD9517=1, ADC=2, CPLD=8 };

// RNW, len[1:0], 13b address, 8b data
void FmcSpi::_writeAD9517(unsigned addr, unsigned val)
{
  unsigned v = (val&0xff)<<16 | ((addr&0xff)<<8) | ((addr&0x1f00)>>8);
  _sp2[AD9517] = v;
  _WAIT;
}

unsigned FmcSpi::_readAD9517(unsigned addr)
{
  unsigned v = (0x80) | ((addr&0x1f00)>>8) | ((addr&0xff)<<8) | (0xFF)<<16;
  _sp2[AD9517] = v;
  _WAIT;

  v = _sp2[0];
  v >>= 16;
  v &= 0xff;
  return v;
}


// WNR, 7b address, 16b data
void FmcSpi::_writeADC(unsigned addr, unsigned val)
{
  unsigned v = (val&0xff)<<16 | (val&0xff00) | (addr&0x7f) | 0x80;
  _sp2[ADC] = v;
  _WAIT;
}

unsigned FmcSpi::_readADC(unsigned addr)
{
  unsigned v = (addr&0x7f) | (0xFFFF)<<8;
  _sp2[ADC] = v;
  _WAIT;

  v = _sp2[0];
  return ((v&0xff0000)>>16) | ((v&0xff00)>>0);
}

// RNW, 7b address, 8b data
void FmcSpi::_writeCPLD(unsigned addr, unsigned val)
{
  unsigned v = (val&0xff)<<8 | (addr&0x7f);
  _sp1[CPLD] = v;
  _WAIT;
}

unsigned FmcSpi::_readCPLD(unsigned addr)
{
  unsigned v = (addr&0x7f) | (0xFF)<<8 | 0x80;
  _sp1[CPLD] = v;
  _WAIT;

  v = _sp1[0];
  return (v>>8)&0xff;
}

int FmcSpi::initSPI()
{
  //  Set SPI mode (CPOL=0, CPHA=0)
  _reg[0xf0] = 0x3;
  return 0;
}

int FmcSpi::cpld_init()
{
  unsigned char fans = 0;
  unsigned syncsrc = 1; // FPGA
#if 1
  unsigned char dirs = 0x1;
  unsigned clksrc = 3;  // external ref
#else
  unsigned char dirs = 0;
  unsigned clksrc = 6;  // internal ref
#endif

  resetSPIclocktree();
  resetSPIadc();

  printf("CPLD   reset  = %x\n",_readCPLD  (0));

  //  Set AD9517 to drive SDO
  _writeAD9517(0,0x99);

  unsigned r0 = (clksrc&0x07)|((syncsrc&0x3)<<3);
  _writeCPLD(0,r0);

  unsigned r1 = (dirs&0xf)|((fans&0xf)<<4);
  _writeCPLD(1,r1);

  _writeCPLD(2,0); // LED off

  uint32_t v = _readCPLD(0);
  if ((v&0xff) != r0) {
    printf("Error verifying reg0 [%x:%x]\n",v&0xff,r0);
    return -1;
  }
  else
    printf("CPLD r0 : %x\n", v);
  
  v = _readCPLD(1);
  if ((v&0xff) != r1) {
    printf("Error verifying reg1 [%x:%x]\n",v&0xff,r1);
    return -1;
  }
  else
    printf("CPLD r1 : %x\n", v);

  return 0;
}

int FmcSpi::resetSPIclocktree()
{
  unsigned v = _readCPLD(0);
  _writeCPLD(0,v|0x20);
  _writeCPLD(0,v&~0x20);
  return 0;
}

int FmcSpi::resetSPIadc()
{
  unsigned v = _readCPLD(0);
  _writeCPLD(0,v|0x40);
  _writeCPLD(0,v&~0x40);
  return 0;
}

int FmcSpi::clocktree_init(unsigned   clocksource, 
                           unsigned   vcotype,
                           TimingType timing)
{
  unsigned v;

  int32_t A,B,P,R;

  switch(timing) {
  case LCLS:
    // 9.996 MHz refclk  (LCLS)
    // Gives 21 samples / 119MHz cycle (294 samples / 8.5MHz cycle)
    A = 4; B = 78;
    P = 6; // P-counter = 32
    R = 10;
    break;
  case LCLSII:
    // 14-6/7 MHz refclk (LCLSII)
    // Gives 2696 samples / 929kHz beam cycle
    A = 21; B = 52;
    P = 6; // P-counter = 32
    R = 10;
    break;
  case EXTERNAL:
  default:
    //  100 MHz refclk
    A = 10; B = 15;
    P = 5; // P-counter = 16
    R = 10;
  }

  printf("clocktree A=%u, B=%u, P=%u, R=%u\n",
         A,B,P,R);

  _writeAD9517( 0x10, 0x7c); //CP 4.8mA, normal op.
  _writeAD9517( 0x11, R);    //R lo
  _writeAD9517( 0x12, 0x00); //R hi
  _writeAD9517( 0x13, A); //A
  _writeAD9517( 0x14, B); //B
  _writeAD9517( 0x14, B); //B lo
  _writeAD9517( 0x15, 0); //B hi
  _writeAD9517( 0x16, P); //presc. DM16
  _writeAD9517( 0x17, 0x84); //STATUS = DLD
  _writeAD9517( 0x19, 0x00);
  _writeAD9517( 0x1A, 0x00); //LD = DLD
  _writeAD9517( 0x1B, 0x00); //REFMON = GND
  //  _writeAD9517( 0x1C, 0x86); //REF1 input
  _writeAD9517( 0x1C, 0x87); //Diff ref input
  _writeAD9517( 0x1D, 0x00); 
  _writeAD9517( 0xF0, 0x02); //out0, safe power down
  _writeAD9517( 0xF1, 0x0C); //out1, lvpecl 960mW
  _writeAD9517( 0xF4, 0x02); //out2, safe power down
  _writeAD9517( 0xF5, 0x0C); //out3, adc, lvpecl 960mW
  _writeAD9517(0x140, 0x01); //out4, sync, pd
  _writeAD9517(0x141, 0x01); //out5, pd
  _writeAD9517(0x142, 0x00); //out6, lvds 1.75mA
  _writeAD9517(0x143, 0x01); //out7, pd
  _writeAD9517(0x190, 0xBC); //div0, clk out, /50 (50MHz)
  _writeAD9517(0x191, 0x00); //div0, clk out, divider used
  _writeAD9517(0x192, 0x00); //div0, clk out, divider to output
  _writeAD9517(0x196, 0x22); //div1, adc, /2
  _writeAD9517(0x197, 0x80); //div1, adc, divider bypassed
  _writeAD9517(0x198, 0x02); //div1, adc, clk to output
  _writeAD9517(0x199, 0x00); //div2.1, /2
  _writeAD9517(0x19A, 0x00); //phase
  _writeAD9517(0x19B, 0x00); //div2.2, /2
  _writeAD9517(0x19C, 0x00); //div2.1 on, div2.2 on
  _writeAD9517(0x19D, 0x00); //div2 dcc on
  _writeAD9517(0x19E, 0x00); //div3.1, /2
  _writeAD9517(0x19F, 0x00); //phase
  _writeAD9517(0x1A0, 0x00); //div3.2, /2
  _writeAD9517(0x1A1, 0x00); //div3.1 on, div3.2 on
  _writeAD9517(0x1A2, 0x00); //div3 dcc on
  _writeAD9517(0x1E0, 0x00); //vco div /2

  _writeAD9517(0x1E1, 0x00); //user external vco and vco divider

  _writeAD9517(0x230, 0x00); //no pwd, no sync
  _writeAD9517(0x232, 0x01); //update 
  
  usleep(100000);
  
  // verify CLK0 PLL status
  v = _readAD9517(0x1F);
  if ((v&0x01)!=0x01) {
  printf("PLL not locked!!!\n");
  return -1;
} else {
  printf("PLL locked!!!\n");
}

  clockWhileSync();

  return 0;
}

void FmcSpi::limitBandwidth(bool b)
{
  unsigned v = _readADC(1);
  if (b)
    v &= ~(1<<8);
  else
    v |=  (1<<8);
  _writeADC(1,v);
}

void FmcSpi::clockWhileSync()
{
  unsigned v = _readADC(1);
  v |= (1<<10);
  _writeADC(1,v);
  
  v = 1;
  _writeADC(6,v);
}

void FmcSpi::_applySync()
{
  //  Apply sync
  uint32_t v = _readCPLD(0);
  printf("applySync-0 %x\n",v);

  v &= ~0x10;
  _writeCPLD(0,v);
  usleep(50000);

  uint32_t q = _readCPLD(0);
  printf("applySync-1 %x\n",q);

  v |= 0x10;
  _writeCPLD(0,v);
  usleep(50000);

  v = _readCPLD(0);
  printf("applySync-2 %x\n",v);
}

int FmcSpi::adc_enable_test(unsigned pattern)
{
  _reg[0xf0] = 0x3;
  _WAIT;

  printf("ADC    partId = %x\n",_readADC   (0));

  while(1) {
    unsigned v;

    v = _readADC(2);
    printf("Active channels : 0x%x\n",v&0xf);

    _writeADC(0x5,pattern);
    v = _readADC(0x5);
    printf("Enable pattern %x\n",v);

    if (v!=pattern) {
      printf("adc_enable_test read pattern %x[%x]\n",
             v,pattern);
      break;
    }

    v = _readADC(1);
    v |= (1<<12);
    _writeADC(1,v);
    v = _readADC(1);
    if (((v>>12)&1)!=1) {
      printf("adc_enable_test read reg1 %x[%x]\n",
             v,(1<<12));
      break;
    }

    _applySync();
    return 0;
  }

  printf("adc_enable_test failed for FMC card %c!\n", _cardId());
  return -1;
}

int FmcSpi::adc_disable_test()
{
  unsigned v = _readADC(1);
  v &= ~(1<<12);
  _writeADC(1,v);

  v = _readADC(1);
  if (((v>>12)&1)!=0) {
    printf("Error confirming disable %x:%x\n",v,0);
    printf("adc_disable_test failed for FMC card %c!\n", _cardId());
    return -1;
  }

  return 0;
}

//
//  front_io_fmc(1) is 156.25MHz calibration signal
//
void FmcSpi::adc_enable_cal()
{
  unsigned r = _readCPLD(1);
  r |= (1<<1);
  _writeCPLD(1,r);
}

void FmcSpi::adc_disable_cal()
{
  unsigned r = _readCPLD(1);
  r &= ~(1<<1);
  _writeCPLD(1,r);
}

void FmcSpi::setAdcMux(unsigned channels)
{
  unsigned v = _readADC(1);

  v &= ~0xf;

  for(unsigned i=0; i<4; i++)
    if (channels & (1<<i)) {
      v |= (3<<2) | i;
      break;
    }

  _writeADC(1,v);
}

void FmcSpi::setAdcMux(bool     interleave,
                       unsigned channels)
{
  unsigned v = _readADC(1);

  v &= ~0xf;

  if (interleave) {
    for(unsigned i=0; i<4; i++)
      if (channels & (1<<i)) {
        v |= (1<<3) | i;
        break;
      }
  }
  _writeADC(1,v);
}

char FmcSpi::_cardId() const
{ return (reinterpret_cast<uint64_t>(this)&0x400ULL) ? 'A':'B'; }

unsigned FmcSpi::get_offset(unsigned channel)
{
  _writeADC(0x0F,channel+1);
  return _readADC(0x21);
}

unsigned FmcSpi::get_gain  (unsigned channel)
{
  _writeADC(0x0F,channel+1);
  return _readADC(0x23);
}

void     FmcSpi::set_offset(unsigned channel, unsigned value)
{
  _writeADC(0x0F,channel+1);
  _writeADC(0x20,value);
  _writeADC(0x10,0x8);
}

void     FmcSpi::set_gain  (unsigned channel, unsigned value)
{
  _writeADC(0x0F,channel+1);
  _writeADC(0x22,value);
  _writeADC(0x10,0x20);
}

