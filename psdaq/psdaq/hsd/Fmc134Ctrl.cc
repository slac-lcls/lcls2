#include "psdaq/hsd/Fmc134Ctrl.hh"

#include <unistd.h>
#include <stdio.h>

using namespace Pds::HSD;

void Fmc134Ctrl::dump()
{
  printf("Present: %c\tPowerGood %c\n", (info&1) ? 'F':'T', (info&2) ? 'T':'F');
  unsigned v = xcvr;
  { printf("xcvr   : %x\t", v);
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
  printf("Status  : %x\n", status);
  printf("ADCvalid: %x\n", adc_val);
  printf("Scramble: %x\n", scramble);
  printf("SWtrig  : %x\n", sw_trigger);
  printf("LMFCcnt : %x\n", lmfc_cnt);
  printf("AlignCh : %x\n", align_char);
  printf("ADC pins: %x / %x\n", adc_pins, adc_pins_r);

#define DUMP_CLK(s,title) {                      \
    test_clksel = s;                             \
    double frq = double(test_clkfrq)/8192.*125.; \
    printf("%s : %f MHz\n", title, frq); }

  DUMP_CLK(0,"REGCLK");
  DUMP_CLK(1,"RX_CLK");
  DUMP_CLK(2,"SYSREF");
  DUMP_CLK(3,"LMKDEV");
  DUMP_CLK(4,"PLLCLK");
  DUMP_CLK(5,"GTREF0.0");
  DUMP_CLK(6,"GTREF0.1");
  DUMP_CLK(7,"GTREF0.2");
  DUMP_CLK(8,"GTREF0.3");

#undef DUMP_CLK
}
