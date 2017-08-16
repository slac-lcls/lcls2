#include <thread>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "spscqueue.h"
#include "mpscqueue.h"
#include "PgpCardMod.h"

#include "spscqueue.h"
#include "mpscqueue.h"

const int NWORKERS = 9;
const int N = 800000;

class MovingAverage
{
public:
    MovingAverage(int n) : index(0), sum(0), N(n), values(N) {}
    int add_value(int value)
    {
        int ret;
        if (index < N) {
            sum += value;
            values[index] = value;
            ret = 0;
        }
        else {
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

void pgp_reader(SPSCQueue& buffer_queue, std::vector<SPSCQueue*>& worker_queues)
{
    int number_of_workers = worker_queues.size();

    int port = 2;
    char dev_name[128];
    snprintf(dev_name, 128, "/dev/pgpcardG3_0_%u", port);
    int fd = open(dev_name, O_RDWR);
    if (fd < 0) {
        std::cout<<"Failed to open pgpcard"<<std::endl;
    }

    PgpCardRx pgp_card;
    pgp_card.model = sizeof(&pgp_card);
    pgp_card.maxSize = 1000000UL;
    pgp_card.pgpLane = port - 1;

    int64_t worker = 0;
    MovingAverage avg_queue_size(number_of_workers);
    for (int i=0; i<N; i++) {
        uint32_t* element = buffer_queue.pop();
        /*
        // read from the pgp card
        pgp_card.data = element;
        int ret = read(fd, &pgp_card, sizeof(pgp_card));
        if (ret <= 0) {
            std::cout<<"Error in reading from pgp card!"<<std::endl;
        }
        else if (ret == pgp_card.maxSize) {
            std::cout<<"Warning! Package size bigger than the maximum size!"<<std::endl;
        }
        */

        float* array = reinterpret_cast<float*>(element);
        int nx = 1;
        int ny = 700;
        for (int j=0; j<nx; j++) {
            for (int k=0; k<ny; k++) {
                array[j*ny + k] = i;
            }
        }

        // load balancing
        SPSCQueue* queue;
        while (true) {
            queue = worker_queues[worker % number_of_workers];
            int queue_size = queue->guess_size();
            // calculate running mean over the last worker queues
            int mean = avg_queue_size.add_value(queue_size);
            if (queue_size * number_of_workers - 10 < mean) {
                break;
            }
            worker++;
        }

        queue->push(element);
        worker++;
    }
}

void worker(SPSCQueue* worker_queue, MPSCQueue& collector, int rank)
{
    int64_t counter = 0;
    while (true) {
        uint32_t* element = worker_queue->pop();
        if (element == nullptr) {
            break;
        }
        /*
        if (rank == 4 || rank == 7) {
            usleep(200);
        }
        */
        // processing goes here
        float* array = reinterpret_cast<float*>(element);
        int nx = 1;
        int ny = 700;
        double sum = 0.0;
        for (int j=0; j<nx; j++) {
            for (int k=0; k<ny; k++) {
                sum += array[j*ny + k];
            }
        }
        *reinterpret_cast<double*>(element) = sum / (nx*ny);

        collector.push(element);
        counter++;
    }
    std::cout<<"Thread "<<rank<<" processed "<<counter<<" events"<<std::endl;
}

void pin_thread(const pthread_t& th, int cpu)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    int rc = pthread_setaffinity_np(th,
                                    sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cout << "Error calling pthread_setaffinity_np: " << rc << "\n";
    }
}

int main()
{
    int queue_size = 2048;

    // pin main thread
    pin_thread(pthread_self(), 1);

    SPSCQueue buffer_queue(queue_size);
    std::vector<SPSCQueue*> worker_queues;
    for (int i=0; i<NWORKERS; i++) {
        worker_queues.push_back(new SPSCQueue(queue_size));
    }
    MPSCQueue collector(queue_size);

    // fill initial queue with buffers
    int64_t buffer_element_size = 5000;
    int64_t buffer_size = queue_size * buffer_element_size;
    std::cout<<"buffer size:  "<<buffer_size * 4 / 1.e9<<" Gb"<<std::endl;
    uint32_t* buffer = new uint32_t[buffer_size];
    for (int i=0; i<queue_size; i++) {
        buffer_queue.push(&buffer[i*buffer_element_size]);
    }

    std::thread pgp_thread(pgp_reader, std::ref(buffer_queue), std::ref(worker_queues));
    pin_thread(pgp_thread.native_handle(), 2);

    std::vector<std::thread> worker_threads;
    for (int i=0; i<NWORKERS; i++) {
        worker_threads.emplace_back(worker, worker_queues[i], std::ref(collector), i);
        pin_thread(worker_threads[i].native_handle(), 3+i);
    }

    std::ofstream out("test.dat");
    auto start = std::chrono::steady_clock::now();
    for (int i=0; i<N; i++) {
        uint32_t* element = collector.pop();

        double value = *reinterpret_cast<double*>(element);
        out<<value<<'\n';
        //std::cout<<"res  "<<value<<std::endl;

        // return buffer to buffer pool
        buffer_queue.push(element);
    }
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // std::cout<< duration / double(N) <<"  ms per message"<<std::endl;
    std::cout<<"Processing rate:  "<<double(N) / duration <<"  kHz"<<std::endl;

    out.close();

    // shutdown worker queues and wait for threads to finish
    for (int i=0; i<NWORKERS; i++) {
        worker_queues[i]->shutdown();
        worker_threads[i].join();
    }
    pgp_thread.join();
    delete [] buffer;
    return 0;
}
