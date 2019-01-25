#include <getopt.h>
#include "pgpdriver.h"

int main(int argc, char* argv[])
{
    AxisG2Device* dev = 0;
    int c;
    while((c = getopt(argc, argv, "d:b:v:")) != EOF) {
        switch(c) {
            case 'b': dev = new AxisG2Device(optarg); break;
            case 'd': dev = new AxisG2Device(strtol(optarg, NULL, 0)); break;
            case 'v': dev->version(strtoul(optarg, NULL, 0)); return 0;
            default:  return -1;
        }
    }

    if (dev)
      dev->status();
}

