#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "mpscqueue.hh"
#include "spscqueue.hh"

#include "PgpCardMod.h"
#include "mpscqueue.hh"
#include "spscqueue.hh"

#include "pds/hdf5/Hdf5Writer.hh"
#include "xtcdata/xtc/Dgram.hh"

using BufferQueue = SPSCQueue<uint32_t*>;

void fex(void* element);

// const int NWORKERS = 9;
// const int N = 800000;
const int N = 500;
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

void pgp_reader(BufferQueue& buffer_queue, SPSCQueue<int>& collector_queue,
                std::vector<BufferQueue*>& worker_input_queues)
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
    // 4-byte units, we think.  should agree with buffer_element_size below
    pgp_card.maxSize = 1000000UL;
    pgp_card.pgpLane = port - 1;

    int64_t worker = 0;
    MovingAverage avg_queue_size(number_of_workers);
    for (int i = 0; i < N; i++) {
        uint32_t* element;
        buffer_queue.pop(element);
        /*
        // read from the pgp card
        pgp_card.data = element;
        unsigned int ret = read(fd, &pgp_card, sizeof(pgp_card));
        std::cout<<"PGP read  "<<ret<<std::endl;
        if (ret <= 0) {
            std::cout << "Error in reading from pgp card!" << std::endl;
        } else if (ret == pgp_card.maxSize) {
            std::cout << "Warning! Package size bigger than the maximum size!" << std::endl;
        }
        */

        // fill arrays by hand and do not use pgp
        float* array = reinterpret_cast<float*>(element);
        int nx = 1;
        int ny = 700;
        for (int j=0; j<nx; j++) {
            for (int k=0; k<ny; k++) {
                array[j*ny + k] = i;
            }
        }

        // load balancing
        BufferQueue* queue;
        while (true) {
            queue = worker_input_queues[worker % number_of_workers];
            int queue_size = queue->guess_size();
            // calculate running mean over the last worker queues
            int mean = avg_queue_size.add_value(queue_size);
            if (queue_size * number_of_workers - 10 < mean) {
                break;
            }
            worker++;
        }

        queue->push(element);
        collector_queue.push(worker% number_of_workers);
        worker++;
    }
}

void worker(BufferQueue* worker_input_queue, BufferQueue* worker_output_queue, int rank)
{
    int64_t counter = 0;
    while (true) {
        uint32_t* element;
        if(!worker_input_queue->pop(element)) {
            break;
        }

        fex(element);

        worker_output_queue->push(element);
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
    int queue_size = 2048;

    // pin main thread
    pin_thread(pthread_self(), 1);

    BufferQueue buffer_queue(queue_size);
    std::vector<BufferQueue*> worker_input_queues;
    std::vector<BufferQueue*> worker_output_queues;
    for (int i = 0; i < NWORKERS; i++) {
        worker_input_queues.push_back(new BufferQueue(queue_size));
        worker_output_queues.push_back(new BufferQueue(queue_size));
    }
    SPSCQueue<int> collector_queue(queue_size);

    // fill initial queue with buffers
    int64_t buffer_element_size = 1000000;
    int64_t buffer_size = queue_size * buffer_element_size;
    std::cout << "buffer size:  " << buffer_size * 4 / 1.e9 << " Gb" << std::endl;
    uint32_t* buffer = new uint32_t[buffer_size];
    for (int i = 0; i < queue_size; i++) {
        buffer_queue.push(&buffer[i * buffer_element_size]);
    }

    std::thread pgp_thread(pgp_reader, std::ref(buffer_queue),
                           std::ref(collector_queue), std::ref(worker_input_queues));
    pin_thread(pgp_thread.native_handle(), 2);

    std::vector<std::thread> worker_threads;
    for (int i = 0; i < NWORKERS; i++) {
        worker_threads.emplace_back(worker, worker_input_queues[i], worker_output_queues[i], i);
        pin_thread(worker_threads[i].native_handle(), 3 + i);
    }

    std::ofstream out("test.dat");
    auto start = std::chrono::steady_clock::now();

    // HDF5File file("/drpffb/weninc/test.h5");
    for (int i = 0; i < N; i++) {
        int worker;
        collector_queue.pop(worker);

        uint32_t* element;
        worker_output_queues[worker]->pop(element);
        //HDF5LevelIter iter(&(((Dgram*)element)->xtc), file);
        //iter.iterate();

        std::cout<<*reinterpret_cast<float*>(element)<<std::endl;
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

    out.close();

    // shutdown worker queues and wait for threads to finish
    for (int i = 0; i < NWORKERS; i++) {
        worker_input_queues[i]->shutdown();
        worker_threads[i].join();
    }
    for (int i = 0; i < NWORKERS; i++) {
        worker_output_queues[i]->shutdown();
    }

    pgp_thread.join();
    delete[] buffer;
    return 0;
}
