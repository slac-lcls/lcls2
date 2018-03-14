#include "mpscqueue.hh"
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <vector>

const int N = 1048576;

void producer(MPSCQueue& queue, uint32_t* buffer, int rank)
{
    int length = N / 8;
    for (int i = 0; i < length; i++) {
        int index = i + rank * length;
        buffer[index] = index;
        queue.push(&buffer[index]);
    }
}

int main()
{
    int nworkers = 8;

    MPSCQueue queue(N);
    uint32_t* buffer = new uint32_t[N];

    std::vector<std::thread> workers;
    for (int i = 0; i < nworkers; i++) {
        workers.emplace_back(producer, std::ref(queue), buffer, i);
    }

    std::ofstream out("test.dat");
    for (int i = 0; i < N; i++) {
        out << queue.pop()[0] << '\n';
        // usleep(10);
    }

    for (int i = 0; i < nworkers; i++) {
        workers[i].join();
    }

    delete[] buffer;
    return 0;
}
