#include <thread>
#include <cstdio>
#include <cstring>
#include <getopt.h>
#include "PGPReader.hh"
#include "AreaDetector.hh"
#include "DigitizerNew.hh"
#include "Worker.hh"
#include "Collector.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/Sequence.hh"
#include <sstream>
#include <iostream>

using namespace XtcData;

void print_usage(){
    printf("Usage: main -P <EB server IP address> -i <Contributor ID> -o <Output XTC dir> -d <Device id> -l <Lane mask>\n");
    printf("e.g.: sudo psdaq/build/drp/main -P 172.21.52.128 -i 0 -o /drpffb/yoon82 -d 0x2032 -l 0xf\n");
}

int main(int argc, char* argv[])
{
    Parameters para;
    int device_id = 0x2031;
    int lane_mask = 0xf;
    int c;
    while((c = getopt(argc, argv, "P:i:o:d:l:")) != EOF) {
        switch(c) {
            case 'P':
                para.eb_server_ip = optarg;
                break;
            case 'i':
                para.contributor_id = atoi(optarg);
                break;
            case 'o':
                para.output_dir = optarg;
                break;
            case 'd':
                device_id = std::stoul(optarg, nullptr, 16);
                break;
            case 'l':
                lane_mask = std::stoul(optarg, nullptr, 16);
                break;
            default:
                print_usage();
                exit(1);
        }
    }
    printf("eb server ip: %s\n", para.eb_server_ip.c_str());
    printf("contributor id: %u\n", para.contributor_id);
    printf("output dir: %s\n", para.output_dir.c_str());

    Factory<Detector> f;
    f.register_type<Digitizer>("Digitizer");
    Detector *d = f.create("Digitizer");
    printf("%p\n", d);

    int num_workers = 2;
    int num_entries = 8192;
    MemPool pool(num_workers, num_entries);
    PGPReader pgp_reader(pool, device_id, lane_mask, num_workers);
    std::thread pgp_thread(&PGPReader::run, std::ref(pgp_reader));

    // start worker threads
    std::vector<std::thread> worker_threads;
    for (int i = 0; i < num_workers; i++) {
        worker_threads.emplace_back(worker, d, std::ref(pool.worker_input_queues[i]),
                                    std::ref(pool.worker_output_queues[i]), i);
    }

    collector(pool, para);

    pgp_thread.join();
    for (int i = 0; i < num_workers; i++) {
        worker_threads[i].join();
    }
}
