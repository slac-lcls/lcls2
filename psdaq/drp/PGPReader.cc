#include <limits.h>
#include <thread>
#include <cstdio>
#include <chrono>
#include <bitset>
#include <fstream>
#include <unistd.h>
#include <zmq.h>
#include "pgpdriver.h"
#include "PGPReader.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/Sequence.hh"

using namespace XtcData;

MovingAverage::MovingAverage(int n) : index(0), sum(0), N(n), values(N, 0) {}
int MovingAverage::add_value(int value)
{
    int& oldest = values[index % N];
    sum += value - oldest;
    oldest = value;
    index++;
    return sum;
}

PGPReader::PGPReader(MemPool& pool, int device_id, int lane_mask, int nworkers) :
    m_dev(device_id),
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

long read_infiniband_counter(const char* counter)
{
    char path[PATH_MAX];
    snprintf(path, PATH_MAX, "/sys/class/infiniband/mlx4_0/ports/1/counters/%s", counter);
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);
    return stol(line);
}

void monitor_pgp(std::atomic<Counters*>& p, MemPool& pool)
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
        int output_queue_size = pool.output_queue.guess_size();
        long port_rcv_data = read_infiniband_counter("port_rcv_data");
        long port_xmit_data = read_infiniband_counter("port_xmit_data");

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t - oldt).count();
        double data_rate = double(new_bytes - old_bytes) / duration;
        double event_rate = double(new_count - old_count) / duration * 1.0e3;
        printf("Event rate %.2f kHz    Data rate  %.2f MB/s\n", event_rate, data_rate);
        int64_t epoch = std::chrono::duration_cast<std::chrono::duration<int64_t>>(
                        std::chrono::system_clock::now().time_since_epoch()).count();

        printf("collector queue %u\n", pool.collector_queue.guess_size());

        // Inifiband counters are divided by 4 (lanes) https://community.mellanox.com/docs/DOC-2751
        double rcv_rate = 4.0*double(port_rcv_data - old_port_rcv_data) / duration;
        double xmit_rate = 4.0*double(port_xmit_data - old_port_xmit_data) / duration;

        int size = snprintf(buffer, 4096,
                R"(["%s",{"time": [%ld], "event_rate": [%f], "data_rate": [%f], "buffer_queue": [%d], "output_queue": [%d], "rcv_rate": [%f], "xmit_rate": [%f]}])",
                hostname, epoch, event_rate, data_rate, buffer_queue_size, output_queue_size, rcv_rate, xmit_rate);
        zmq_send(socket, buffer, size, 0);
        // printf("%s\n", buffer);
        old_bytes = new_bytes;
        old_count = new_count;
        old_port_rcv_data = port_rcv_data;
        old_port_xmit_data = port_xmit_data;
    }
}

void PGPReader::run()
{
    // start monitoring thread
    Counters c1, c2;
    Counters* counter = &c2;
    std::atomic<Counters*> p(&c1);
    std::thread monitor_thread(monitor_pgp, std::ref(p), std::ref(m_pool));

    int64_t event_count = 0;
    int64_t total_bytes_received = 0;
    while (true) {
        DmaBuffer* buffer = m_dev.read();
        if (buffer->size > RX_BUFFER_SIZE) {
            printf("ERROR: Buffer overflow, pgp message %d kB is bigger than RX_BUFFER_SIZE %d kB\n",
                    buffer->size/1024, RX_BUFFER_SIZE/1024);
        }
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

            send_to_worker(pebble);

            /*
            switch (transition_id) {
                case 0:
                    send_to_worker(pebble);
                    break;

                case 2:
                    // FIXME
                    // send_all_workers(pebble);
                    send_to_worker(pebble);
                    break;
                default:
                    printf("Unknown transition %d\n", transition_id);
                    break;
            }
            */
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
