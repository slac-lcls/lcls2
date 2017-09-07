#include <thread>
#include <vector>
#include <iostream>
#include <fstream>
#include "spscqueue.hh"
#include "mpscqueue.hh"

const int NWORKERS = 8;
const int N = 1048576;

void pgp_reader(SPSCQueue& buffer_queue, std::vector<SPSCQueue*>& worker_queues)
{
    int number_of_workers = worker_queues.size();

    int64_t worker = 0;
    for (int i=0; i<N; i++) {
        uint32_t* element = buffer_queue.pop();
        //std::cout<<"producer:  "<<i<<'\n';

        *element = i;

        SPSCQueue* queue = worker_queues[worker % number_of_workers];
        queue->push(element);
        worker++;
        // std::cout<<"producer:  "<<*element<<std::endl;
    }
}

void worker(SPSCQueue* worker_queue, MPSCQueue& collector)
{
    for (int i=0; i<N/NWORKERS; i++) {
        uint32_t* element = worker_queue->pop();

        *element = *element + 1;

        collector.push(element);
    }
}

int main()
{
    int queue_size = 32;

    SPSCQueue buffer_queue(queue_size);
    std::vector<SPSCQueue*> worker_queues;
    for (int i=0; i<NWORKERS; i++) {
        worker_queues.push_back(new SPSCQueue(queue_size));
    }
    MPSCQueue collector(queue_size);

    // fill initial queue with buffers
    uint32_t* buffer = new uint32_t[queue_size];
    for (int i=0; i<queue_size; i++) {
        buffer_queue.push(&buffer[i]);
    }

    std::thread pgp_thread(pgp_reader, std::ref(buffer_queue), std::ref(worker_queues));
    std::vector<std::thread> worker_threads;
    for (int i=0; i<NWORKERS; i++) {
        worker_threads.emplace_back(worker, worker_queues[i], std::ref(collector));
    }

    std::ofstream out("test.dat");
    for (int i=0; i<N; i++) {
        uint32_t* element = collector.pop();
        out<<*element<<'\n';
        // std::cout<<"res  "<<*element<<std::endl;

        // return buffer to buffer pool
        buffer_queue.push(element);
    }
    out.close();


    for (int i=0; i<NWORKERS; i++) {
        worker_threads[i].join();
    }
    pgp_thread.join();
}
