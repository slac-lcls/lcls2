#include "psdaq/hsd/Fmc134Ctrl.hh"
#include "psdaq/hsd/Fmc134Cpld.hh"

#include <unistd.h>
#include <stdio.h>

using namespace Pds::HSD;

void Fmc134Ctrl::dump()
{
  printf("Present: %c\tPowerGood %c\n", (info&1) ? 'F':'T', (info&2) ? 'T':'F');
  { unsigned v = xcvr;
    printf("xcvr   : %x\t", v);
    if (v&0x01) printf(" rxrst");
    if (v&0x00002) printf(" sysref_sync");
    if (v&0x00010) printf(" align_en");
    if (v&0x00100) printf(" dfe_agc_hold");
    if (v&0x00200) printf(" dfe_lf_hold");
    if (v&0x00400) printf(" dfe_tap_hold");
    if (v&0x00800) printf(" lpm_gc_hold");
    if (v&0x01000) printf(" lpm_hf_hold");
    if (v&0x02000) printf(" lpm_lf_hold");
    if (v&0x04000) printf(" lpm_os_hold");
    if (v&0x08000) printf(" rx_os_hold");
    if (v&0x10000) printf(" rx_cdr_hold");
    if (v&0x20000) printf(" dfe_tap_ovrd");
    printf("\n"); }
  { unsigned v = status;
    printf("Status  : %x\t", v);
    printf("~PMAresetDone %x\t", (~v)&0xffff);
    printf("~RxByteAligned %x\t", (~v>>16)&0xf);
    printf("~QPLLlock %x\t", (~v>>20)&0xf); 
    printf("\n"); }
  printf("ADCvalid: %x\n", adc_val);
  printf("Scramble: %x\n", scramble);
  printf("SWtrig  : %x\n", sw_trigger);
  printf("LMFCcnt : %x\n", lmfc_cnt);
  printf("AlignCh : %x\n", align_char);
  printf("ADC pins: %x / %x\n", adc_pins, adc_pins_r);
  { unsigned v = adc_pins;
    printf("\tSYNC %u\tNCO %x\n", v&1, (v>>8)&0xff); }
  { unsigned v = adc_pins_r;
    printf("\tOR %x\tCALSTAT %x\tFIREFLY %x\n", (v&0xff), (v>>16)&0x3, (v>>18)&1); }

#define DUMP_CLK(s,title) {                      \
    test_clksel = s;                             \
    double frq = double(test_clkfrq)/8192.*125.; \
    printf("%s : %f MHz\n", title, frq); }

  DUMP_CLK(0,"REGCLK");
  DUMP_CLK(1,"RX_CLK");
  DUMP_CLK(2,"SYSREF");
  DUMP_CLK(3,"LMKDEV");
  // DUMP_CLK(4,"PLLCLK");
  // DUMP_CLK(5,"GTREF0.0");
  // DUMP_CLK(6,"GTREF0.1");
  // DUMP_CLK(7,"GTREF0.2");
  // DUMP_CLK(8,"GTREF0.3");

#undef DUMP_CLK
}

class FMC134Offset {
public:
  enum { AddrCtrl=0 };
};

#define unitapi_write_register(unit,addr,val) reinterpret_cast<uint32_t*>(this)[addr] = val
#define unitapi_read_register(unit,addr,pval) *(pval) = reinterpret_cast<uint32_t*>(this)[addr]
#define unitapi_sleep_ms(tms) usleep(tms*1000)
static const int32_t UNITAPI_OK = 0;
static const int32_t FMC134_ERR_ADC_INIT = 1;
static const int32_t FMC134_ERR_OK = 0;

void    Fmc134Ctrl::remote_sync ()
{
        // Assert SYNC pin
        unitapi_write_register(fmc_unit,  FMC134Offset::AddrCtrl+0x08, 0x1); 

        // Release SYNC
        unitapi_write_register(fmc_unit,  FMC134Offset::AddrCtrl+0x08, 0x0); 
}

int32_t Fmc134Ctrl::default_init(Fmc134Cpld& cpld, unsigned mode)
{
  int32_t rc = UNITAPI_OK;
        uint32_t dword = 0;

        // Enable Scrambling in JESD204B core
        //        unitapi_write_register(fmc_unit,  FMC134Offset::AddrCtrl+0x04, 0x1); 
        unitapi_write_register(fmc_unit,  FMC134Offset::AddrCtrl+0x04, 0x0); 
        if(rc!=UNITAPI_OK)     return rc;

        // Try a different align char (default=0xfc)
        unitapi_write_register(fmc_unit,  FMC134Offset::AddrCtrl+0x07, 0xfc); 
        if(rc!=UNITAPI_OK)     return rc;

        // Configure the ADC0 and ADC1 to generate PRBS23
        cpld.config_prbs(0x3);

        // Stop hold TAP values, CDR, and AGCs
        unitapi_write_register(fmc_unit,  FMC134Offset::AddrCtrl+0x1, 0x0); 
        if(rc!=UNITAPI_OK)     return rc;

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Configure Tranceiver
        // Assert Transceiver Reset
        unitapi_write_register(fmc_unit,  FMC134Offset::AddrCtrl+0x01, 0x1); 
        if(rc!=UNITAPI_OK)     return rc;
        // Release Tranceiver Reset
        unitapi_write_register(fmc_unit,  FMC134Offset::AddrCtrl+0x01, 0x0);
        if(rc!=UNITAPI_OK)     return rc;

        // Wait for MGTs to adapt
        unitapi_sleep_ms(100);

        // Hold TAP values, CDR, and AGCs
        unitapi_write_register(fmc_unit,  FMC134Offset::AddrCtrl+0x1, 0x1FF00); 
        if(rc!=UNITAPI_OK)     return rc;

        //Disable PRBS
        cpld.config_prbs(mode);

        // Wait for QPLLs to lock
        unitapi_sleep_ms(100);

        // Assert DIV2 Reset
        unitapi_write_register(fmc_unit,  FMC134Offset::AddrCtrl+0x01, 0x1FF02); 
        if(rc!=UNITAPI_OK)     return rc;
        // Release DIV2 Reset
        unitapi_write_register(fmc_unit,  FMC134Offset::AddrCtrl+0x01, 0x1FF00);
        if(rc!=UNITAPI_OK)     return rc;

        unitapi_read_register(fmc_unit,   FMC134Offset::AddrCtrl+0x02, &dword);
        if(rc!=UNITAPI_OK)     return rc;
        dword &= 0xF00000; 
        if (dword != 0xF00000) {
                printf("QPLLs NOT LOCKED!\n");
        } else {
                printf("QPLLs are locked.\n");
        }

        // Enable transceiver alignment
        unitapi_write_register(fmc_unit,  FMC134Offset::AddrCtrl+0x01, 0x1FF10);
        if(rc!=UNITAPI_OK)     return rc;
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Check for JESD ADC to be stable
        unitapi_read_register(fmc_unit,   FMC134Offset::AddrCtrl + 0x02, &dword);
        if(rc!=UNITAPI_OK)     return rc;

        if (((dword >> 16) & 0x3) == 0x3)
                printf("ADC0 Aligned\n");
        else {
                printf("ADC0 Failed Bit Alignment!\n");
                return FMC134_ERR_ADC_INIT;
        }

        if (((dword >> 18) & 0x3) == 0x3)
                printf("ADC1 Aligned\n");
        else {
                printf("ADC1 Failed Bit Alignment!\n");
                return FMC134_ERR_ADC_INIT;
        }

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Check for JESD multiframe alignment
        unitapi_read_register(fmc_unit,   FMC134Offset::AddrCtrl+0x03, &dword);
        if(rc!=UNITAPI_OK)     return rc;

        if (dword == 0xF) {
                printf("Initial Lane Alignment Complete\n");
                return FMC134_ERR_OK;
        }
        else {
                printf("\n\nADC Initial Lane Alignment Failed!\n");
                printf("reg7 = 0x%X\n\n", dword);
                return FMC134_ERR_ADC_INIT;
        }
}
