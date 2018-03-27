#ifndef MPSCQUEUE_H
#define MPSCQUEUE_H

#include <atomic>
#include <condition_variable>

// Multiple producer single consumer queue
// push is threadsafe and can be called by multiple threads
// while pop can only be called by a single thread and blocks until there is at least one element in
// the queue
class MPSCQueue
{
public:
    MPSCQueue(int capacity);
    void push(uint32_t* value);
    uint32_t* pop();
    bool is_empty();
    ~MPSCQueue();

private:
    alignas(64) std::atomic<int64_t> cursor;
    alignas(64) std::atomic<int64_t> write_index;
    alignas(64) std::atomic<int64_t> read_index;
    std::mutex _mutex;
    std::condition_variable _condition;
    int64_t buffer_mask;
    uint32_t** ringbuffer;
};

#endif // MPSCQUEUE_H
