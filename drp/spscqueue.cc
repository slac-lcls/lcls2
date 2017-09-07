#include <cassert>
#include "spscqueue.hh"

SPSCQueue::SPSCQueue(int capacity) : write_index(0), read_index(0), terminate(false)
{
    buffer.resize(capacity);
    buffer_mask = capacity - 1;
}

void SPSCQueue::push(uint32_t* value)
{
    int64_t index = write_index.load();
    buffer[index & buffer_mask] = value;
    int64_t next = index+ 1;
    write_index.store(next); // std::memory_order_release
    // signal consumer
    if (index == read_index.load(std::memory_order_acquire)) {
        std::unique_lock<std::mutex> lock(_mutex);
        _condition.notify_one();
    }
}

uint32_t* SPSCQueue::pop()
{
    int64_t index = read_index.load();

    // Queue is empty
    if (index == write_index.load()) { //std::memory_order_acquire
        std::unique_lock<std::mutex> lock(_mutex);
        _condition.wait(lock, [this]{return !is_empty() || terminate.load();});
        if (terminate && is_empty()) {
            return nullptr;
        }
    }

    uint32_t* value = buffer[index & buffer_mask];
    int64_t next = index + 1;
    read_index.store(next, std::memory_order_release);
    return value;
}

bool SPSCQueue::is_empty()
{
    return read_index.load() == write_index.load();
}

int SPSCQueue::guess_size()
{
    int ret = write_index.load(std::memory_order_acquire) - read_index.load(std::memory_order_acquire);
    return std::max(0, ret);
}

void SPSCQueue::shutdown()
{
    {
        std::unique_lock<std::mutex> lock(_mutex);
        terminate.store(true);
    }
    _condition.notify_one();
}
