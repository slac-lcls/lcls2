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

// these parameters must agree with the server side
size_t maxSize = sizeof(MyDgram);

using namespace XtcData;
using json = nlohmann::json;
using namespace Pds::Eb;

void print_usage(){
    printf("Usage: drp -p <partition>  -o <Output XTC dir> -d <Device id> -l <Lane mask> -D <Detector type>\n");
    printf("e.g.: sudo psdaq/build/drp/drp -p 1 -o /drpffb/username -d 0x2032 -l 0xf -D Digitizer\n");
}

void join_collection(Parameters& para)
{
    Collection collection("drp-tst-acc06", para.partition, "drp");
    collection.connect();
    std::cout << "cmstate:\n" << collection.cmstate.dump(4) << std::endl;

    std::string id = std::to_string(collection.id());
    // std::cout << "DRP: " << id << std::endl;
    para.tPrms.id = collection.cmstate["drp"][id]["drp_id"];

    const unsigned numPorts    = MAX_DRPS + MAX_TEBS + MAX_MEBS + MAX_MEBS;
    const unsigned tebPortBase = TEB_PORT_BASE + numPorts * para.partition;
    const unsigned drpPortBase = DRP_PORT_BASE + numPorts * para.partition;
    const unsigned mebPortBase = MEB_PORT_BASE + numPorts * para.partition;

    para.tPrms.port = std::to_string(drpPortBase + para.tPrms.id);

    uint64_t builders = 0;
    for (auto it : collection.cmstate["teb"].items()) {
        unsigned    tebId   = it.value()["teb_id"];
        std::string address = it.value()["connect_info"]["infiniband"];
        std::cout << "TEB: " << tebId << "  " << address << std::endl;
        builders |= 1ul << tebId;
        para.tPrms.addrs.push_back(address);
        para.tPrms.ports.push_back(std::string(std::to_string(tebPortBase + tebId)));
    }
    para.tPrms.builders = builders;

    if (collection.cmstate.find("meb") != collection.cmstate.end()) {
        for (auto it : collection.cmstate["meb"].items()) {
            unsigned    mebId   = it.value()["meb_id"];
            std::string address = it.value()["connect_info"]["infiniband"];
            std::cout << "MEB: " << mebId << "  " << address << std::endl;
            para.mPrms.addrs.push_back(address);
            para.mPrms.ports.push_back(std::string(std::to_string(mebPortBase + mebId)));
        }
    }
}

int main(int argc, char* argv[])
{
    Parameters para;
    para.partition = 0;
    int lane_mask = 0xf;
    std::string detector_type;
    int c;
    while((c = getopt(argc, argv, "p:o:l:D:")) != EOF) {
        switch(c) {
            case 'p':
                para.partition = std::stoi(optarg);
                break;
            case 'o':
                para.output_dir = optarg;
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

    // event builder
    para.tPrms = { /* .addrs         = */ { },
                   /* .ports         = */ { },
                   /* .ifAddr        = */ nullptr,
                   /* .port          = */ { },
                   /* .id            = */ 0,
                   /* .builders      = */ 0,
                   /* .duration      = */ BATCH_DURATION,
                   /* .maxBatches    = */ MAX_BATCHES,
                   /* .maxEntries    = */ MAX_ENTRIES,
                   /* .maxInputSize  = */ maxSize,
                   /* .maxResultSize = */ maxSize,
                   /* .core          = */ { 11 + 0,
                                            12 },
                   /* .verbose       = */ 0 };

    para.mPrms = { /* .addrs         = */ { },
                   /* .ports         = */ { },
                   /* .id            = */ 0,
                   /* .maxEvents     = */ 8,    //mon_buf_cnt,
                   /* .maxEvSize     = */ 1024, //mon_buf_size,
                   /* .maxTrSize     = */ 1024, //mon_trSize,
                   /* .verbose       = */ 0 };

    join_collection(para);
    printf("output dir: %s\n", para.output_dir.c_str());

    Factory<Detector> f;
    f.register_type<Digitizer>("Digitizer");
    f.register_type<AreaDetector>("AreaDetector");
    Detector* d = f.create(detector_type.c_str());

    int num_workers = 2;
    int num_entries = 8192;
    MemPool pool(num_workers, num_entries);
    // TODO: This should be moved to configure when the lane_mask is known.
    PGPReader pgp_reader(pool, lane_mask, num_workers);
    std::thread pgp_thread(&PGPReader::run, std::ref(pgp_reader));
    pin_thread(pgp_thread.native_handle(), 1);

    Pds::Eb::EbContributor ebCtrb(para.tPrms);
    Pds::Eb::MonContributor* meb = nullptr;
    if (para.mPrms.addrs.size() != 0) {
        meb = new Pds::Eb::MonContributor(para.mPrms);
    }

    // start performance monitor thread
    std::thread monitor_thread(monitor_func, std::ref(pgp_reader.get_counters()),
                               std::ref(pool), std::ref(ebCtrb));

    // start worker threads
    std::vector<std::thread> worker_threads;
    for (int i = 0; i < num_workers; i++) {
        worker_threads.emplace_back(worker, d, std::ref(pool.worker_input_queues[i]),
                                    std::ref(pool.worker_output_queues[i]), i);
        pin_thread(worker_threads[i].native_handle(), 2 + i);
    }

    collector(pool, para, ebCtrb, meb);

    pgp_thread.join();
    for (int i = 0; i < num_workers; i++) {
        worker_threads[i].join();
    }

    // shutdown monitor thread
    // counter->total_bytes_received = -1;
    //p.exchange(counter, std::memory_order_release);
    // monitor_thread.join();
}
