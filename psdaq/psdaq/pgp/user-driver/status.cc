#include <getopt.h>
#include "pgpdriver.h"

int main(int argc, char* argv[])
{
    int device_id = 2031;
    int c;
    while((c = getopt(argc, argv, "d:")) != EOF) {
        switch(c) {
            case 'd': device_id = strtol(optarg, NULL, 0); break;
        }
    }

    AxisG2Device dev(device_id);
    dev.status();
}

