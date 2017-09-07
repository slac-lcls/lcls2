#include <chrono>
#include <thread>
#include <iostream>
#include "mpscqueue.hh"

MPSCQueue::MPSCQueue(int capacity) : cursor(0L), write_index(0L)
{
    read_index = 0L;
    buffer_mask = capacity - 1;
    ringbuffer = new uint32_t*[capacity];
}

void MPSCQueue::push(uint32_t* value)
{
     int64_t slot = cursor.fetch_add(1L, std::memory_order_acq_rel);
     ringbuffer[slot & buffer_mask] = value;
     while (slot != write_index.load());
     int64_t next = slot + 1;
     write_index.store(next);
     // signal consumer
     if (slot == read_index.load(std::memory_order_acquire)) {
         std::unique_lock<std::mutex> lock(_mutex);
         _condition.notify_one();
     }
}

uint32_t * MPSCQueue::pop()
{
    int64_t index = read_index.load();

    // Queue is empty
    if (index == write_index.load()) {
        std::unique_lock<std::mutex> lock(_mutex);
        _condition.wait(lock, [this]{return !is_empty();});
    }

    uint32_t* value = ringbuffer[index & buffer_mask];
    int64_t next = index + 1;
    read_index.store(next, std::memory_order_release);
    return value;
}

bool MPSCQueue::is_empty()
{
    return read_index.load() == write_index.load();
}

MPSCQueue::MPSCQueue::~MPSCQueue()
{
    delete [] ringbuffer;
}
