#include <thread>
#include <cstdio>
#include <cstring>
#include <getopt.h>
#include "PGPReader.hh"

int main(int argc, char* argv[]) 
{
    char dev_name[128];
    unsigned num_lanes;
    int c;
    while((c = getopt(argc, argv, "aP:L:D")) != EOF) {
        switch(c) {
            case 'P':
                strcpy(dev_name, optarg);
                break;
            case 'L':
                num_lanes = strtoul(optarg, NULL, 0);
                break;
            case 'D':
                break;
        }
    }

   
    int num_workers = 2; 
    MemPool pool(num_workers, 10000);
    PGPReader pgp_reader(pool, num_lanes, num_workers);
    std::thread pgp_thread(&PGPReader::run, std::ref(pgp_reader));
    pgp_thread.join();
    
}
