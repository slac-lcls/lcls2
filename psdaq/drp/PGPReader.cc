#include <thread>
#include <cstdio>
#include <chrono>
#include <bitset>
#include <unistd.h>
#include "pgpdriver.h"
#include "PGPReader.hh"

#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/Sequence.hh"

using namespace XtcData;

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

MovingAverage::MovingAverage(int n) : index(0), sum(0), N(n), values(N, 0) {}
int MovingAverage::add_value(int value)
{
    int& oldest = values[index % N];
    sum += value - oldest;
    oldest = value;
    index++;
    return sum;
}

PGPReader::PGPReader(MemPool& pool, int lane_mask, int nworkers) :
    m_dev(0x2032),
    m_pool(pool),
    m_avg_queue_size(nworkers)
{
    std::bitset<32> bs(lane_mask);
    m_nlanes = bs.count();
    m_last_complete = 0;

    m_worker = 0;
    m_nworkers = nworkers;

    m_buffer_mask = m_pool.num_entries - 1;
    m_dev.init(&m_pool.dma);
    m_dev.setup_lanes(lane_mask);
}

PGPData* PGPReader::process_lane(DmaBuffer* buffer)
{
    Transition* event_header = reinterpret_cast<Transition*>(buffer->virt);
    int j = event_header->evtCounter & m_buffer_mask;
    PGPData* p = &m_pool.pgp_data[j];
    p->buffers[buffer->dest] = buffer;

    // set bit in lane mask for lane
    p->buffer_mask |= (1 << buffer->dest);
    p->counter++;
    if (p->counter == m_nlanes) {
        if (event_header->evtCounter != (m_last_complete + 1)) {
            printf("Jump in complete l1Count %d -> %u\n",
                   m_last_complete, event_header->evtCounter);
            // FIXME clean up broken events and return dma indices
        }
        m_last_complete = event_header->evtCounter;
        return p;
    }
    else {
        return nullptr;
    }
}

void PGPReader::send_to_worker(Pebble* pebble_data)
{
    PebbleQueue* queue;
    // load balanching to find worker to send the event to
    while (true) {
        queue = &m_pool.worker_input_queues[m_worker % m_nworkers];
        int queue_size = queue->guess_size();
        // calculate running mean over the worker queues
        int mean = m_avg_queue_size.add_value(queue_size);
        if (queue_size * m_nworkers - 5 < mean) {
            break;
        }
        m_worker++;
    }
    queue->push(pebble_data);
    m_pool.collector_queue.push(m_worker % m_nworkers);
    m_worker++;
}

void PGPReader::send_all_workers(Pebble* pebble)
{
    for (int i=0; i<m_nworkers; i++) {
        m_pool.worker_input_queues[i].push(pebble);
    }
    // only pass on event from worker 0 to collector
    m_pool.collector_queue.push(0);
}

void monitor_pgp(std::atomic<Counters*>& p)
{
    Counters* c = p.load(std::memory_order_acquire);
    int64_t old_bytes = c->total_bytes_received;
    int64_t old_count = c->event_count;
    auto t = std::chrono::steady_clock::now();

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

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t - oldt).count();
        double data_rate = double(new_bytes - old_bytes) / duration;
        double event_rate = double(new_count - old_count) / duration * 1.0e3;
        printf("Event rate %.2f kHz    Data rate  %.2f MB/s\n", event_rate, data_rate);
        old_bytes = new_bytes;
        old_count = new_count;
    }
}

void PGPReader::run()
{
    // start monitoring thread
    Counters c1, c2;
    Counters* counter = &c2;
    std::atomic<Counters*> p(&c1);
    std::thread monitor_thread(monitor_pgp, std::ref(p));

    int64_t event_count = 0;
    int64_t total_bytes_received = 0;
    while (true) {
        DmaBuffer* buffer = m_dev.read();
        total_bytes_received += buffer->size;
        PGPData* pgp = process_lane(buffer);
        if (pgp) {
            // get first set bit to find index of the first lane
            int index = __builtin_ffs(pgp->buffer_mask) - 1;
            Transition* event_header = reinterpret_cast<Transition*>(pgp->buffers[index]->virt);
            TransitionId::Value transition_id = event_header->seq.service();
            //printf("Complete evevent:  Transition id %d pulse id %lu  event counter %u\n", 
            //        transition_id, event_header->seq.pulseId().value(), event_header->evtCounter);
            Pebble* pebble;
            m_pool.pebble_queue.pop(pebble);
            pebble->pgp_data = pgp;
            switch (transition_id) {
                case 0:
                    send_to_worker(pebble);
                    break;

                case 2:
                    send_all_workers(pebble);
                    break;
                default:
                    printf("Unknown transition %d\n", transition_id);
                    break;
            }
            event_count += 1;

            counter->event_count = event_count;
            counter->total_bytes_received = total_bytes_received;
            counter = p.exchange(counter, std::memory_order_release);
        }
    }
    // shutdown monitor thread
    counter->total_bytes_received = -1;
    p.exchange(counter, std::memory_order_release);
    monitor_thread.join();
}
