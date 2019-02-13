#include <limits.h>
#include <unistd.h>
#include <time.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <cstddef>
#include <cstdio>
#include <sys/types.h>
#include "AxisDriver.h"
#include "drp.hh"
#include <zmq.h>

MemPool::MemPool(int num_workers, int num_entries) :
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

    fd = open("/dev/datadev_1", O_RDWR);
    if (fd < 0) {
        printf("Error opening /dev/datadev_1\n");
    }
    uint32_t dmaCount, dmaSize;
    dmaBuffers = dmaMapDma(fd, &dmaCount, &dmaSize);
    if (dmaBuffers == nullptr) {
        std::cout<<"Error calling dmaMapDma!!\n";
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
    snprintf(path, PATH_MAX, "/sys/class/infiniband/mlx5_0/ports/1/counters/%s", counter);
    std::ifstream in(path);
    if (in.is_open()) {
        std::string line;
        std::getline(in, line);
        return stol(line);
    }
    else {
        return 0;
    }
}

void monitor_func(const Parameters& para, std::atomic<Counters*>& p,
                  MemPool& pool, Pds::Eb::TebContributor& ebCtrb)
{
    void* context = zmq_ctx_new();
    void* socket = zmq_socket(context, ZMQ_PUB);
    zmq_connect(socket, "tcp://psmetric04:5559");
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
        // int buffer_queue_size = pool.dma.buffer_queue.guess_size();
        long port_rcv_data = read_infiniband_counter("port_rcv_data");
        long port_xmit_data = read_infiniband_counter("port_xmit_data");

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t - oldt).count();
        double seconds = duration / 1.0e6;
        double data_rate = double(new_bytes - old_bytes) / seconds;
        double event_rate = double(new_count - old_count) / seconds;
        // Inifiband counters are divided by 4 (lanes) https://community.mellanox.com/docs/DOC-2751
        double rcv_rate = 4.0*double(port_rcv_data - old_port_rcv_data) / seconds;
        double xmit_rate = 4.0*double(port_xmit_data - old_port_xmit_data) / seconds;

        int size = snprintf(buffer, 4096, "drp_event_rate,host=%s,partition=%d %f",
                            hostname, para.partition, event_rate);
        zmq_send(socket, buffer, size, 0);

        size = snprintf(buffer, 4096, "drp_data_rate,host=%s,partition=%d %f",
                        hostname, para.partition, data_rate);
        zmq_send(socket, buffer, size, 0);

        size = snprintf(buffer, 4096, "drp_xmit_rate,host=%s,partition=%d %f",
                        hostname, para.partition, xmit_rate);
        zmq_send(socket, buffer, size, 0);

        size = snprintf(buffer, 4096, "drp_rcv_rate,host=%s,partition=%d %f",
                        hostname, para.partition, rcv_rate);
        zmq_send(socket, buffer, size, 0);

        old_bytes = new_bytes;
        old_count = new_count;
        old_port_rcv_data = port_rcv_data;
        old_port_xmit_data = port_xmit_data;
    }
}
