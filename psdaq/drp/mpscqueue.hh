/**
 * @file
 * @brief MPSCQueue, a ring buffer of uint32_t pointers for multiple producers and one consumer.
 */
#ifndef MPSCQUEUE_H
#define MPSCQUEUE_H

#include <atomic>
#include <condition_variable>

// Multiple producer single consumer queue
// push is threadsafe and can be called by multiple threads
// while pop can only be called by a single thread and blocks until there is at least one element in
// the queue
/** Ring buffer of uint32_t pointers. push() may be called from several threads; pop() from one thread only and blocks until an element is available (per the header comment). There is no check for a full queue. */
class MPSCQueue
{
public:
    /** Allocate a ring of capacity pointers; capacity is used through a mask, so it must be a power of 2. */
    MPSCQueue(int capacity);
    /** Claim the next slot, store value in it, wait until earlier pushes have been published, then publish it and wake the consumer if it was waiting on this slot. */
    void push(uint32_t* value);
    /** Return the oldest element, blocking while the queue is empty. */
    uint32_t* pop();
    /** Return true if all published elements have been popped. */
    bool is_empty();
    /** Free the ring. */
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
