#include <thread>
#include <cstdio>
#include <getopt.h>
#include <unistd.h>
#include <limits.h>
#include <sstream>
#include <iostream>
#include "PGPReader.hh"
#include "AreaDetector.hh"
#include "Digitizer.hh"
#include "Worker.hh"
#include "Collector.hh"
#include "psdaq/service/Collection.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/Sequence.hh"
#include <zmq.h>

using namespace XtcData;
using json = nlohmann::json;

void print_usage(){
    printf("Usage: drp -p <partition>  -o <Output XTC dir> -d <Device id> -l <Lane mask> -D <Detector type>\n");
    printf("e.g.: sudo psdaq/build/drp/drp -p 1 -o /drpffb/username -d 0x2032 -l 0xf -D Digitizer\n");
}

void join_collection(Parameters& para)
{
    Collection collection("drp-tst-acc06", para.partition, "drp");
    collection.connect();
    std::cout<<"cmstate:\n"<<collection.cmstate.dump(4) << std::endl;
    std::string id = std::to_string(collection.id());
    std::cout<<id<<std::endl;
    int drp_id = collection.cmstate["drp"][id]["drp_id"];
    // only works with single eb for now. FIXME for multiple eb servers
    for (auto it : collection.cmstate["eb"].items()) {
        para.eb_server_ip = it.value()["connect_info"]["infiniband"];
    }
    para.contributor_id = drp_id;
}

int main(int argc, char* argv[])
{
    Parameters para;
    int device_id = 0x2031;
    int lane_mask = 0xf;
    std::string detector_type;
    int c;
    while((c = getopt(argc, argv, "p:o:d:l:D:")) != EOF) {
        switch(c) {
            case 'p':
                para.partition = std::stoi(optarg);
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
    join_collection(para);
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
    unsigned timeout = 5;
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
