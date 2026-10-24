/**
 * @file
 * @brief SPSCQueue, a fixed-capacity ring buffer for one producer and one consumer, with blocking and non-blocking reads.
 */
#ifndef SPSCQUEUE_H
#define SPSCQUEUE_H

#include <atomic>
#include <mutex>
#include <vector>
#include <condition_variable>
#include <cstdio>

#include "psdaq/service/fast_monotonic_clock.hh"

/** Fixed-capacity ring buffer for one producer thread and one consumer thread. push() never checks for a full queue; the read functions can block on a condition variable, and shutdown() releases blocked readers. */
template <typename T>
class SPSCQueue
{
    using ms_t = std::chrono::milliseconds;
public:
    /** Allocate capacity slots; throws a C string (after printing a message) if capacity is not a power of 2. */
    SPSCQueue(int capacity) : m_terminate(false), m_write_index(0), m_read_index(0)
    {
        if ((capacity & (capacity - 1)) != 0) {
            // Need a better solution: don't want to include stdio.h in an hh file
            fprintf(stderr, "SPSCQueue capacity must be a power of 2, got %d\n", capacity);
            throw "SPSCQueue capacity must be a power of 2";
        };
        m_ring_buffer.resize(capacity);
        m_capacity = capacity;
        m_buffer_mask = capacity - 1;
    }

    /** Copying is disabled. */
    SPSCQueue(const SPSCQueue&) = delete;
    /** Copy assignment is disabled. */
    void operator=(const SPSCQueue&) = delete;

    /** Move constructor: take over the slots, indices, capacity and mask of d; the terminate flag starts cleared. */
    SPSCQueue(SPSCQueue&& d) noexcept
    {
        m_terminate.store(false);
        m_write_index.store(d.m_write_index.load());
        m_read_index.store(d.m_read_index.load());
        m_ring_buffer = std::move(d.m_ring_buffer);
        m_capacity = d.m_capacity;
        m_buffer_mask = d.m_buffer_mask;
    }

    // Write an item to the back of the queue
    // Note that this does not check that the queue is full first!
    /** Write value at the back and wake a reader if the queue was empty. Does not check whether the queue is full (per the code comment). */
    void push(T value)
    {
        int64_t index = m_write_index.load(std::memory_order_relaxed);
        m_ring_buffer[index & m_buffer_mask] = value;
        int64_t next = index + 1;
        m_write_index.store(next, std::memory_order_release);
        // avoid reordering of the write_index store and the read_index load
        asm volatile("mfence" ::: "memory");
        // signal consumer that queue is no longer empty
        if (index == m_read_index.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_condition.notify_one();
        }
    }

    // blocking read from queue
    /** Pop the front element into value, blocking while the queue is empty. Returns false, without popping, if the queue was empty on entry and shutdown() has been called (even if an element arrived meanwhile); otherwise true. */
    bool popW(T& value)
    {
        int64_t index = m_read_index.load(std::memory_order_relaxed);

        // check if queue is empty
        if (index == m_write_index.load(std::memory_order_acquire)) {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_condition.wait(lock, [this] {
                return !is_empty() || m_terminate.load(std::memory_order_acquire);
            });
            if (m_terminate.load(std::memory_order_acquire)) { // && is_empty()) {
                return false;           // Empty and terminating
            }
        }

        // Not empty
        value = m_ring_buffer[index & m_buffer_mask];
        int64_t next = index + 1;
        m_read_index.store(next, std::memory_order_release);
        return true;
    }

    // blocking read from queue with ms timeout
    /** Like popW(T&), but waits at most msTmo milliseconds; returns false on timeout or if terminating while still empty. */
    bool popW(T& value, unsigned msTmo)
    {
        int64_t index = m_read_index.load(std::memory_order_relaxed);

        // check if queue is empty
        if (index == m_write_index.load(std::memory_order_acquire)) {
            const auto tmo{std::chrono::milliseconds(msTmo)};
            std::unique_lock<std::mutex> lock(m_mutex);
            if (!m_condition.wait_for(lock, tmo, [this] {
                    return !is_empty() || m_terminate.load(std::memory_order_acquire);
                })) {
                return false;           // Timed out
            }
            if (m_terminate.load(std::memory_order_acquire) && is_empty()) {
                return false;           // Empty and terminating
            }
        }

        // Not empty
        value = m_ring_buffer[index & m_buffer_mask];
        int64_t next = index + 1;
        m_read_index.store(next, std::memory_order_release);
        return true;
    }

    // blocking read from queue with polling for the 1st ms before blocking
    /** Pop the front element into value, polling for up to 1 ms and then blocking with popW(T&). Returns false only if popW() does. */
    bool pop(T& value)
    {
        int64_t index = m_read_index.load(std::memory_order_relaxed);

        // check if queue is empty
        auto t0 = Pds::fast_monotonic_clock::now();
        while (index == m_write_index.load(std::memory_order_acquire)) {
            auto t1 = Pds::fast_monotonic_clock::now();
            if (t1 - t0 >= ms_t(1)) {
                return popW(value);
            }
        }

        value = m_ring_buffer[index & m_buffer_mask];
        int64_t next = index + 1;
        m_read_index.store(next, std::memory_order_release);
        return true;
    }

    // non blocking read from queue
    /** Pop the front element into value without blocking; returns false if the queue is empty. */
    bool try_pop(T& value)
    {
        int64_t index = m_read_index.load(std::memory_order_relaxed);

        // check if queue is empty
        if (index == m_write_index.load(std::memory_order_acquire)) {
            return false;
        }
        value = m_ring_buffer[index & m_buffer_mask];
        int64_t next = index + 1;
        m_read_index.store(next, std::memory_order_release);
        return true;
    }

    // Inspect the front of the queue returning empty status
    /** Copy the front element into value without removing it; returns false if the queue is empty. */
    bool peek(T& value) const
    {
        int64_t index = m_read_index.load(std::memory_order_relaxed);

        // check if queue is empty
        if (index == m_write_index.load(std::memory_order_acquire)) {
            return false;
        }
        value = m_ring_buffer[index & m_buffer_mask];
        return true;
    }

    // Inspect the front of the queue
    /** Return a reference to the slot at the read position (not checked for emptiness). */
    T& front()
    {
        int64_t index = m_read_index.load(std::memory_order_relaxed);
        return m_ring_buffer[index & m_buffer_mask];
    }

    // Inspect the front of the queue
    /** Return a const reference to the slot at the read position (not checked for emptiness). */
    const T& front() const
    {
        int64_t index = m_read_index.load(std::memory_order_relaxed);
        return m_ring_buffer[index & m_buffer_mask];
    }

    // Inspect the nth entry from the front, for walking the queue in a dump.
    // Valid for n < guess_size() when called from the consumer thread.
    const T& at(int64_t n) const
    {
        int64_t index = m_read_index.load(std::memory_order_relaxed);
        return m_ring_buffer[(index + n) & m_buffer_mask];
    }

    // Inspect the back of the queue
    /** Return a reference to the slot at the write position, the next one push() will fill. */
    T& back()
    {
        int64_t index = m_write_index.load(std::memory_order_relaxed);
        return m_ring_buffer[index & m_buffer_mask];
    }

    // Inspect the back of the queue
    /** Return a const reference to the slot at the write position, the next one push() will fill. */
    const T& back() const
    {
        int64_t index = m_write_index.load(std::memory_order_relaxed);
        return m_ring_buffer[index & m_buffer_mask];
    }

    // Check whether the queue is empty
    /** Return true if the read and write positions are equal. */
    bool is_empty() const
    {
        return m_read_index.load(std::memory_order_acquire) ==
               m_write_index.load(std::memory_order_acquire);
    }

    // Wait for the queue to become nonempty
    /** Block until the queue is not empty; returns false if shutdown() was called while it is still empty, else true. */
    bool wait_for_nonempty() const
    {
        if (is_empty()) {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_condition.wait(lock, [this] {
                return !is_empty() || m_terminate.load(std::memory_order_acquire);
            });
            if (m_terminate.load(std::memory_order_acquire) && is_empty()) {
                return false;
            }
        }
        return true;
    }

    // Return the occupancy of the queue
    /** Return the number of queued elements (write minus read position, plus the capacity if negative). */
    int guess_size() const
    {
        int ret = m_write_index.load(std::memory_order_acquire) -
                  m_read_index.load(std::memory_order_acquire);
        if (ret < 0) {
            ret += m_capacity;
        }
        return ret;
    }

    // Return the capacity of the queue
    /** Return the capacity. */
    size_t size() const
    {
        return m_ring_buffer.size();
    }

    // Shut down the queue by releasing the lock
    /** Set the terminate flag and wake one blocked reader. */
    void shutdown()
    {
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_terminate.store(true, std::memory_order_release);
        }
        m_condition.notify_one();
    }

    // Reset the state of the queue to allow it to be reused
    /** Clear the terminate flag and reset both positions to 0. */
    void startup()
    {
        m_terminate.store(false);
        m_write_index.store(0);
        m_read_index.store(0);
    }

private:
    mutable std::mutex m_mutex;
    mutable std::condition_variable m_condition;
    std::atomic<bool> m_terminate;
    int64_t m_buffer_mask, m_capacity;
    std::vector<T> m_ring_buffer;
    alignas(64) std::atomic<int64_t> m_write_index;
    alignas(64) std::atomic<int64_t> m_read_index;
    char _pad[64 - sizeof(std::atomic<int64_t>)];
};

#endif // SPSCQUEUE_H
