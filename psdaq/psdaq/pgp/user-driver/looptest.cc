#include <getopt.h>
#include "pgpdriver.h"

void show_usage(const char* p) {
    printf("Usage: %s [options]\n", p);
    printf("Options:\n"
           "\t-T <code>  Trigger at fixed interval\n"
           "\t   0x00-0x0f = 156   - 10    MHz  [  1 - 16 clks]\n"
           "\t   0x10-0x1f =  39   -  2.5  MHz  [  4  - 64 clks]\n"
           "\t   0x20-0x2f =  10   -  0.6  MHz  [ 16  - 256 clks]\n"
           "\t   0x31-0x3f =   2.5 -  0.15 MHz  [ 64  - 1024 clks]\n"
           "\t   0x41-0x4f = 600   -  38   kHz  [256  - 4k clks]\n"
           "\t   0x51-0x5f = 152   -  9.5  kHz  [  1k - 16k clks]\n"
           "\t   0x61-0x6f =  38   -  2.4  kHz  [  4k - 64k clks]\n"
           "\t   0x71-0x7f =   9.5 -  0.6  kHz  [ 16k - 256k clks]\n"
           "\t-L <lanes> Bit mask of enabled lanes\n"
           "\t-s <words> Tx size in 32b words\n");
}


int main(int argc, char* argv[])
{
    unsigned lanes = 1;
    unsigned op_code = 0x71;
    unsigned size = 32;
    int c;
    while((c = getopt(argc, argv, "T:L:s:SD:")) != EOF) {
        switch(c) {
            case 'T': op_code = strtoul(optarg, NULL, 0); break;
            case 'L': lanes  = strtoul(optarg, NULL, 0); break;
            case 's': size   = strtoul(optarg, NULL, 0); break;
            default: show_usage(argv[0]); return 0;
        }
    }

    AxisG2Device dev;
    dev.loop_test(lanes, size, op_code);
}
