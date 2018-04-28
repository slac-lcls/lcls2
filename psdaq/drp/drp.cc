#include "drp.hh"

MemPool::MemPool(int num_workers, int num_entries) :
    dma(num_entries, RX_BUFFER_SIZE),
    pgp_data(num_entries),
    pebble_queue(num_entries),
    collector_queue(num_entries),
    output_queue(num_entries),
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
