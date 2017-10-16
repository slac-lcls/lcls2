#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>
#include <sstream>
#include <memory>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "spscqueue.hh"
#include "PgpCardMod.h"

/* Instruction on how to run DTI

~weaver/l2rel/build/pdsapp/bin/x86_64-rhel7-opt/dti_simple -a 10.0.2.104 -u 3 -f 1,2

to control trigger rate:

~weaver/l2rel/build/pdsapp/bin/x86_64-rhel7-opt/xpm_simple -a 10.0.2.102 -r 2 -e

-e enables backpressure

-r sets rate, 6 is 1Hz, 5 is 10Hz, 4 is 100Hz, 3 1000Hz, 2 10kHz, 1 71kHz, 0 1MHz


~weaver/l2si/software/app/pgpcardG3/xLoopTest -d /dev/pgpcardG3_0_1 -T 0x80 -e 0 -s 1500 -S
~weaver/l2si/software/app/pgpcardG3/xLoopTest -d /dev/pgpcardG3_0_4 -T 0x80 -e 3 -s 1500 -S

*/

using BufferQueue = SPSCQueue<uint32_t*>;

struct EventHeader {
    uint64_t pulseId;
    uint64_t timeStamp;
    uint32_t trigTag;
    uint32_t l1Count;
    unsigned rawSamples:24;
    unsigned channelMask:8;
    uint32_t reserved;
};

const int N = 1000000;
const int NWORKERS = 1;

class MovingAverage
{
public:
    MovingAverage(int n) : index(0), sum(0), N(n), values(N, 0) {}
    int add_value(int value)
    {
        int& oldest = values[index % N];
        sum += value - oldest;
        oldest = value;
        index++;
        return sum;
    }

private:
    int64_t index;
    int sum;
    int N;
    std::vector<int> values;
};

void monitor_pgp(std::atomic<int64_t>& total_bytes_received,
                      std::atomic<int64_t>& event_count)
{
    int64_t old_bytes = total_bytes_received.load(std::memory_order_relaxed);;
    int64_t old_count = event_count.load(std::memory_order_relaxed);
    auto t = std::chrono::steady_clock::now();
    while (1) {
        sleep(1);
        auto oldt = t;
        t = std::chrono::steady_clock::now();

        int64_t new_bytes = total_bytes_received.load(std::memory_order_relaxed);
        if (new_bytes == -1) {
            break;
        }
        int64_t new_count = event_count.load(std::memory_order_relaxed);

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t - oldt).count();
        double data_rate = double(new_bytes - old_bytes) / duration;
        double event_rate = double(new_count - old_count) / duration * 1.0e3;
        printf("Event rate %.2f kHz    Data rate  %.2f MB/s\n", event_rate, data_rate);

        old_bytes = new_bytes;
        old_count = new_count;

    }
}

void pgp_reader(BufferQueue& buffer_queue, SPSCQueue<int>& collector_queue, std::vector<BufferQueue>& worker_input_queues)
{
    int number_of_workers = worker_input_queues.size();

    int ports[] = {2, 4};
    int fds[2];
    int nchannels = 2;
    for (int c=0; c<nchannels; c++) {
        std::stringstream ss;
        ss<<"/dev/pgpcardG3_0_"<<ports[c];
        std::string dev_name = ss.str();
        fds[c] = open(dev_name.c_str(), O_RDWR);
        if (fds[c] < 0) {
            std::cout << "Failed to open pgpcard" << std::endl;
        }
    }

    int64_t worker = 0;
    MovingAverage avg_queue_size(number_of_workers);

    std::atomic<int64_t> total_bytes_received(0L);
    std::atomic<int64_t> event_count(0L);
    std::thread monitor_thread(monitor_pgp, std::ref(total_bytes_received), std::ref(event_count));

    for (int i = 0; i < N; i++) {
        uint32_t* element;
        buffer_queue.pop(element);

        PgpCardRx pgp_card;
        pgp_card.model = sizeof(&pgp_card);
        // 4-byte units, should agree with buffer_element_size below
        pgp_card.maxSize = 1000000UL;
        pgp_card.data = element;
        uint64_t bytes_received = 0UL;
        uint64_t pulse_id;
        for (int c=0; c<nchannels; c++) {
            unsigned int ret = read(fds[c], &pgp_card, sizeof(pgp_card));
            bytes_received += ret*4;
            if (ret <= 0) {
                std::cout << "Error in reading from pgp card!" << std::endl;
            } else if (ret == pgp_card.maxSize) {
                std::cout << "Warning! Package size bigger than the maximum size!" << std::endl;
            }

            // confirm that the message from all channels have the same pulse id
            EventHeader* event_header = reinterpret_cast<EventHeader*>(element);
            if (c == 0) {
                pulse_id = event_header->pulseId;
            }
            else {
                if (event_header->pulseId != pulse_id) {
                    std::cout<<"non matching pulse ids\n";
                }
            }
            pgp_card.data += ret;
        }
        // update pgp metrics
        uint64_t temp = event_count.load(std::memory_order_relaxed) + 1;
        event_count.store(temp, std::memory_order_relaxed);

        temp = total_bytes_received.load(std::memory_order_relaxed) + bytes_received;
        total_bytes_received.store(temp, std::memory_order_relaxed);

        // load balancing
        BufferQueue* queue;
        while (true) {
            queue = &worker_input_queues[worker % number_of_workers];
            int queue_size = queue->guess_size();
            // calculate running mean over the worker queues
            int mean = avg_queue_size.add_value(queue_size);
            if (queue_size * number_of_workers - 5 < mean) {
                break;
            }
            worker++;
        }

        queue->push(element);
        collector_queue.push(worker % number_of_workers);
        worker++;
    }

    // shutdown monitor thread
    total_bytes_received.store(-1, std::memory_order_relaxed);
    monitor_thread.join();
}

void worker(BufferQueue& worker_input_queue, BufferQueue& worker_output_queue, int rank)
{
    int64_t counter = 0;
    while (true) {
        uint32_t* element;
        if (!worker_input_queue.pop(element)) {
            break;
        }

        //EventHeader* event_header = reinterpret_cast<EventHeader*>(element);
        //printf("[Event]: pulseId %016llu  timestamp %016llu  l1Count %08x  channelMask %02x raw samples %d\n",
        //       event_header->pulseId, event_header->timeStamp, event_header->l1Count,
        //       event_header->channelMask, event_header->rawSamples);
        //uint32_t* payload = reinterpret_cast<uint32_t*>(event_header+1);
        // std::cout<<payload[4]<<std::endl;

        worker_output_queue.push(element);
        counter++;
    }
    std::cout << "Thread " << rank << " processed " << counter << " events" << std::endl;
}

void pin_thread(const pthread_t& th, int cpu)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    int rc = pthread_setaffinity_np(th, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cout << "Error calling pthread_setaffinity_np: " << rc << "\n";
    }
}

int main()
{
    int queue_size = 16384;

    // pin main thread
    pin_thread(pthread_self(), 1);

    BufferQueue buffer_queue(queue_size);
    std::vector<BufferQueue> worker_input_queues;
    std::vector<BufferQueue> worker_output_queues;
    for (int i = 0; i < NWORKERS; i++) {
        worker_input_queues.emplace_back(BufferQueue(queue_size));
        worker_output_queues.emplace_back(BufferQueue(queue_size));
    }
    
    // FIXME
    // BufferQueue* test = (BufferQueue*)malloc(NWORKERS*sizeof(BufferQueue));
    // for (int i=0; i<NWORKERS; i++) {
    //    new(void*(test[i])) BufferQueue(queue_size);
    //}

    SPSCQueue<int> collector_queue(queue_size);

    // buffer size in elements of 4 byte units
    int64_t buffer_element_size = 100000;
    int64_t buffer_size = queue_size * buffer_element_size;
    std::cout << "buffer size:  " << buffer_size * 4 / 1.e9 << " GB" << std::endl;
    uint32_t* buffer = new uint32_t[buffer_size];
    for (int i = 0; i < queue_size; i++) {
        buffer_queue.push(&buffer[i * buffer_element_size]);
    }

    std::thread pgp_thread(pgp_reader, std::ref(buffer_queue), std::ref(collector_queue),
                           std::ref(worker_input_queues));
    pin_thread(pgp_thread.native_handle(), 2);

    std::vector<std::thread> worker_threads;
    for (int i = 0; i < NWORKERS; i++) {
        worker_threads.emplace_back(worker, std::ref(worker_input_queues[i]),
                                    std::ref(worker_output_queues[i]), i);
        pin_thread(worker_threads[i].native_handle(), 3 + i);
    }

    std::ofstream out("test.dat");
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < N; i++) {
        int worker;
        collector_queue.pop(worker);

        uint32_t* element;
        worker_output_queues[worker].pop(element);

        // return buffer to buffer pool
        buffer_queue.push(element);
    }
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // std::cout<< duration / double(N) <<"  ms per message"<<std::endl;
    std::cout << "Processing rate:  " << double(N) / duration << "  kHz" << std::endl;
    std::cout << "Processing time:  " << duration / 1000.0 << "  s" << std::endl;
    out.close();

    // shutdown worker queues and wait for threads to finish
    for (int i = 0; i < NWORKERS; i++) {
        worker_input_queues[i].shutdown();
        worker_threads[i].join();
    }
    for (int i = 0; i < NWORKERS; i++) {
        worker_output_queues[i].shutdown();
    }

    // buffer_queue.shutdown();
    pgp_thread.join();
    delete[] buffer;
    return 0;
}
