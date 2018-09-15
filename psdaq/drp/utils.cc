#include <limits.h>
#include <unistd.h>
#include <time.h>
#include <fstream>
#include <cstddef>
#include <cstdio>
#include <sys/types.h>
#include "drp.hh"
#include "Collector.hh"
#include <zmq.h>

MemPool::MemPool(int num_workers, int num_entries) :
    dma(num_entries, RX_BUFFER_SIZE),
    pgp_data(num_entries),
    pebble_queue(num_entries),
    collector_queue(num_entries),
    num_entries(num_entries),
    pebble(num_entries)
{
    for (int i = 0; i < num_workers; i++) {
        worker_input_queues.emplace_back(PebbleQueue(num_entries));
        worker_output_queues.emplace_back(PebbleQueue(num_entries));
    }

    for (int i = 0; i < num_entries; i++) {
        pgp_data[i].counter = 0;
        pgp_data[i].buffer_mask = 0;
        pebble_queue.push(&pebble[i]);
    }
}

void pin_thread(const pthread_t& th, int cpu)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    int rc = pthread_setaffinity_np(th, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        printf("Error calling pthread_setaffinity_np: %d\n ", rc);
    }
}

long read_infiniband_counter(const char* counter)
{
    char path[PATH_MAX];
    snprintf(path, PATH_MAX, "/sys/class/infiniband/mlx4_0/ports/1/counters/%s", counter);
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);
    return stol(line);
}

void monitor_func(std::atomic<Counters*>& p, MemPool& pool, Pds::Eb::EbContributor& ebCtrb)
{
    void* context = zmq_ctx_new();
    void* socket = zmq_socket(context, ZMQ_PUB);
    zmq_connect(socket, "tcp://psdev7b:5559");
    char buffer[4096];
    char hostname[HOST_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);

    Counters* c = p.load(std::memory_order_acquire);
    int64_t old_bytes = c->total_bytes_received;
    int64_t old_count = c->event_count;
    auto t = std::chrono::steady_clock::now();

    long old_port_rcv_data = read_infiniband_counter("port_rcv_data");
    long old_port_xmit_data = read_infiniband_counter("port_xmit_data");

    while(1) {
        sleep(1);
        auto oldt = t;
        t = std::chrono::steady_clock::now();

        Counters* c = p.load(std::memory_order_acquire);
        int64_t new_bytes = c->total_bytes_received;
        if (new_bytes == -1) {
            break;
        }
        int64_t new_count = c->event_count;
        int buffer_queue_size = pool.dma.buffer_queue.guess_size();
        long port_rcv_data = read_infiniband_counter("port_rcv_data");
        long port_xmit_data = read_infiniband_counter("port_xmit_data");

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t - oldt).count();
        double data_rate = double(new_bytes - old_bytes) / duration;
        double event_rate = double(new_count - old_count) / duration * 1.0e3;
        printf("Event rate:      %.2f kHz  Data rate  %.2f MB/s\nCollector queue:  %u  Used batches    %d\n",
                event_rate, data_rate,
                pool.collector_queue.guess_size(),
                ebCtrb.inFlightCnt());

        // Inifiband counters are divided by 4 (lanes) https://community.mellanox.com/docs/DOC-2751
        double rcv_rate = 4.0*double(port_rcv_data - old_port_rcv_data) / duration;
        double xmit_rate = 4.0*double(port_xmit_data - old_port_xmit_data) / duration;

        time_t rawtime;
        tm* timeinfo;
        time (&rawtime);
        timeinfo = localtime(&rawtime);

        char time_buffer[80];
        strftime(time_buffer, 80, "%Y-%m-%d %H:%M:%S", timeinfo);

        int size = snprintf(buffer, 4096,
                R"({"host": "%s", "x": "%s", "data": {"event_rate": [%f], "data_rate": [%f], "rcv_rate": [%f], "xmit_rate": [%f], "buffer_queue": [%d], "used_batches": [%d]}})",
                            hostname, time_buffer, event_rate, data_rate, rcv_rate, xmit_rate, buffer_queue_size, ebCtrb.inFlightCnt());

        /*
 "event_rate": [%f], "data_rate": [%f], "buffer_queue": [%d], "output_queue": [%d], "rcv_rate": [%f], "xmit_rate": [%f]}])",
                hostname, epoch, event_rate, data_rate, buffer_queue_size, output_queue_size, rcv_rate, xmit_rate);
                */
        zmq_send(socket, buffer, size, 0);
        // printf("%s\n", buffer);
        old_bytes = new_bytes;
        old_count = new_count;
        old_port_rcv_data = port_rcv_data;
        old_port_xmit_data = port_xmit_data;
    }
}
