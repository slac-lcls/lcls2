#include <thread>
#include <cstdio>
#include <cstring>
#include <getopt.h>
#include "PGPReader.hh"
#include "AreaDetector.hh"
#include "Digitizer.hh"
#include "Worker.hh"
#include "Collector.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/Sequence.hh"
#include <sstream>
#include <iostream>

using namespace XtcData;

void print_usage(){
    printf("Usage: drp -P <EB server IP address> -i <Contributor ID> -o <Output XTC dir> -d <Device id> -l <Lane mask> -D <Detector type>\n");
    printf("e.g.: sudo psdaq/build/drp/drp -P 172.21.52.128 -i 0 -o /drpffb/yoon82 -d 0x2032 -l 0xf -D Digitizer\n");
}

int main(int argc, char* argv[])
{
    Parameters para;
    int device_id = 0x2031;
    int lane_mask = 0xf;
    std::string detector_type;
    int c;
    while((c = getopt(argc, argv, "P:i:o:d:l:D:")) != EOF) {
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
            case 'D':
                detector_type = optarg;
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
    f.register_type<AreaDetector>("AreaDetector");
    Detector* d = f.create(detector_type.c_str());

    int num_workers = 2;
    // int num_entries = 131072;
    int num_entries = 8192;
    MemPool pool(num_workers, num_entries);
    PGPReader pgp_reader(pool, device_id, lane_mask, num_workers);
    std::thread pgp_thread(&PGPReader::run, std::ref(pgp_reader));
    pin_thread(pgp_thread.native_handle(), 1);

    // event builder
    Pds::StringList peers;
    peers.push_back(para.eb_server_ip);
    Pds::StringList ports;
    ports.push_back("32768");
    Pds::Eb::EbLfClient myEbLfClient(peers, ports);
    MyBatchManager myBatchMan(myEbLfClient, para.contributor_id);
    unsigned timeout = 10;
    int ret = myEbLfClient.connect(para.contributor_id, timeout,
                                   myBatchMan.batchRegion(), 
                                   myBatchMan.batchRegionSize());
    if (ret) {
        printf("ERROR in connecting to event builder!!!!\n");
    }

    // start performance monitor thread
    std::thread monitor_thread(monitor_func, std::ref(pgp_reader.get_counters()),
                               std::ref(pool), std::ref(myBatchMan));

    // start worker threads
    std::vector<std::thread> worker_threads;
    for (int i = 0; i < num_workers; i++) {
        worker_threads.emplace_back(worker, d, std::ref(pool.worker_input_queues[i]),
                                    std::ref(pool.worker_output_queues[i]), i);
        pin_thread(worker_threads[i].native_handle(), 2 + i);
    }

    collector(pool, para, myBatchMan);

    pgp_thread.join();
    for (int i = 0; i < num_workers; i++) {
        worker_threads[i].join();
    }

    // shutdown monitor thread
    // counter->total_bytes_received = -1;
    //p.exchange(counter, std::memory_order_release);
    // monitor_thread.join();
}
