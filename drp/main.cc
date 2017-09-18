#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "spscqueue.hh"

#include "PgpCardMod.h"
#include "mpscqueue.hh"
#include "spscqueue.hh"

#include "pds/hdf5/Hdf5Writer.hh"
#include "xtcdata/xtc/Dgram.hh"


/* Instruction on how to run DTI

~weaver/l2rel/build/pdsapp/bin/x86_64-rhel7-opt/dti_simple -a 10.0.2.104

to control trigger rate:

~weaver/l2rel/build/pdsapp/bin/x86_64-rhel7-opt/xpm_simple -a 10.0.2.102 -r 6 -e

-e enables backpressure

-r sets rate, 6 is 1Hz, 5 is 10Hz, 4 is 100Hz, 3 1000Hz, 2 10kHz, 1 71kHz, 0 1MHz

*/

using BufferQueue = SPSCQueue<uint32_t*>;

void fex(void* element);

struct EventHeader {
  uint64_t pulseId;
  uint64_t timeStamp;
  uint32_t trigTag;
  uint32_t l1Count;
  unsigned rawSamples:24;
  unsigned channelMask:8;
  uint32_t reserved;
};

const int N = 10000;
const int NWORKERS = 8;

class MovingAverage
{
public:
    MovingAverage(int n) : index(0), sum(0), N(n), values(N)
    {
    }
    int add_value(int value)
    {
        int ret;
        if (index < N) {
            sum += value;
            values[index] = value;
            ret = 0;
        } else {
            int& oldest = values[index % N];
            sum += value - oldest;
            oldest = value;
            ret = sum;
        }
        index++;
        return ret;
    }

private:
    int64_t index;
    int sum;
    int N;
    std::vector<int> values;
};

void pgp_reader(BufferQueue& buffer_queue, SPSCQueue<int>& collector_queue, std::vector<BufferQueue>& worker_input_queues)
{
    int number_of_workers = worker_input_queues.size();

    int port = 2;
    char dev_name[128];
    snprintf(dev_name, 128, "/dev/pgpcardG3_0_%u", port);
    int fd = open(dev_name, O_RDWR);
    if (fd < 0) {
        std::cout << "Failed to open pgpcard" << std::endl;
    }

    PgpCardRx pgp_card;
    pgp_card.model = sizeof(&pgp_card);
    // 4-byte units, should agree with buffer_element_size below
    pgp_card.maxSize = 1000000UL;
    pgp_card.pgpLane = port - 1;

    int64_t worker = 0;
    MovingAverage avg_queue_size(number_of_workers);

    uint64_t total_bytes_received = 0UL;
    uint64_t events_received = 0UL;
    auto start_time = std::chrono::steady_clock::now();
    for (int i = 0; i < N; i++) {
        uint32_t* element;
        buffer_queue.pop(element);

        // read from the pgp card
        pgp_card.data = element;
        unsigned int ret = read(fd, &pgp_card, sizeof(pgp_card));
        events_received++;
        total_bytes_received += ret*4;
        // std::cout<<"PGP read  "<<ret<<std::endl;
        if (ret <= 0) {
            std::cout << "Error in reading from pgp card!" << std::endl;
        } else if (ret == pgp_card.maxSize) {
            std::cout << "Warning! Package size bigger than the maximum size!" << std::endl;
        }

        /*
        // fill arrays by hand and do not use pgp
        int* array = reinterpret_cast<int*>(element);
        int nx = 1;
        int ny = 700;
        for (int j = 0; j < nx; j++) {
            for (int k = 0; k < ny; k++) {
                array[j * ny + k] = i;
            }
        }
        */
        // load balancing
        BufferQueue* queue;
        while (true) {
            queue = &worker_input_queues[worker % number_of_workers];
            int queue_size = queue->guess_size();
            // calculate running mean over the last worker queues
            int mean = avg_queue_size.add_value(queue_size);
            if (queue_size * number_of_workers - 10 < mean) {
                break;
            }
            worker++;
        }

        queue->push(element);
        collector_queue.push(worker % number_of_workers);
        worker++;
    }
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout<<"PGP input:  "<<total_bytes_received / duration / 1000.0<< " MB/s\n";
    std::cout<<events_received<< "  events received\n";
}

void worker(BufferQueue& worker_input_queue, BufferQueue& worker_output_queue, int rank)
{
    int64_t counter = 0;
    while (true) {
        uint32_t* element;
        if (!worker_input_queue.pop(element)) {
            break;
        }

        EventHeader* event_header = reinterpret_cast<EventHeader*>(element);
        //printf("[Event]: pulseId %016llu  timestamp %016llu  l1Count %08x  channelMask %02x raw samples %d\n",
        //       event_header->pulseId, event_header->timeStamp, event_header->l1Count,
        //       event_header->channelMask, event_header->rawSamples);
        uint32_t* payload = reinterpret_cast<uint32_t*>(event_header+1);
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

    // HDF5File file("/drpffb/weninc/test.h5");
    for (int i = 0; i < N; i++) {
        int worker;
        collector_queue.pop(worker);

        uint32_t* element;
        worker_output_queues[worker].pop(element);
        // HDF5LevelIter iter(&(((Dgram*)element)->xtc), file);
        // iter.iterate();

        //std::cout << *reinterpret_cast<int*>(element) << std::endl;
        // double value = *reinterpret_cast<double*>(element);
        // out<<value<<'\n';
        // std::cout<<"res  "<<value<<std::endl;

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
