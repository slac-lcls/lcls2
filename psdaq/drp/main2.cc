#include <thread>
#include <cstdio>
#include <cstring>
#include <getopt.h>
#include "PGPReader.hh"
#include "AreaDetector.hh"
#include "Worker.hh"
#include "Collector.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/Sequence.hh"

using namespace XtcData;

int main(int argc, char* argv[])
{
    Parameters para;
    int c;
    while((c = getopt(argc, argv, "P:i:")) != EOF) {
        switch(c) {
            case 'P':
                para.eb_server_ip = optarg;
                break;
            case 'i':
                para.contributor_id = atoi(optarg);
                break;
        }
    }
    para.output_dir = "/drpffb/yoon82";
    printf("eb server ip: %s\n", para.eb_server_ip.c_str());
    printf("contributor id: %u\n", para.contributor_id);
    printf("output dir: %s\n", para.output_dir.c_str());

    Factory<Detector> f;
    f.register_type<AreaDetector>("AreaDetector");
    Detector *d = f.create("AreaDetector");
    printf("%p\n", d);

    int num_workers = 2;
    MemPool pool(num_workers, 65536);
    int lane_mask = 0xf;
    int device_id = 0x2031;
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
