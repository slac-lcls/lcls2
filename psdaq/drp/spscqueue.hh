#ifndef SPSCQUEUE_H
#define SPSCQUEUE_H

#include <atomic>
#include <mutex>
#include <vector>
#include <condition_variable>

#include "psdaq/service/fast_monotonic_clock.hh"

template <typename T>
class SPSCQueue
{
    using us_t = std::chrono::microseconds;
public:
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

    SPSCQueue(const SPSCQueue&) = delete;
    void operator=(const SPSCQueue&) = delete;

    SPSCQueue(SPSCQueue&& d) noexcept
    {
        m_terminate.store(false);
        m_write_index.store(d.m_write_index.load());
        m_read_index.store(d.m_read_index.load());
        m_ring_buffer = std::move(d.m_ring_buffer);
        m_capacity = d.m_capacity;
        m_buffer_mask = d.m_buffer_mask;
    }

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
            std::unique_lock<std::mutex> lock(m_mutex);
            m_condition.notify_one();
        }
    }

    // blocking read from queue
    bool popW(T& value)
    {
        int64_t index = m_read_index.load(std::memory_order_relaxed);

        // check if queue is empty
        if (index == m_write_index.load(std::memory_order_acquire)) {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_condition.wait(lock, [this] {
                return !is_empty() || m_terminate.load(std::memory_order_acquire);
            });
            if (m_terminate.load(std::memory_order_acquire) && is_empty()) {
                return false;
            }
        }

        value = m_ring_buffer[index & m_buffer_mask];
        int64_t next = index + 1;
        m_read_index.store(next, std::memory_order_release);
        return true;
    }

    // blocking read from queue with polling for the 1st N us before blocking
    bool pop(T& value)
    {
        int64_t index = m_read_index.load(std::memory_order_relaxed);

        // check if queue is empty
        auto t0 = Pds::fast_monotonic_clock::now(CLOCK_MONOTONIC);
        while (index == m_write_index.load(std::memory_order_acquire)) {
            auto t1 = Pds::fast_monotonic_clock::now(CLOCK_MONOTONIC);
            auto dt = std::chrono::duration_cast<us_t>(t1 - t0).count();
            if (dt > 1000) {
                return popW(value);
            }
        }

        value = m_ring_buffer[index & m_buffer_mask];
        int64_t next = index + 1;
        m_read_index.store(next, std::memory_order_release);
        return true;
    }

    // non blocking read from queue
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

    // non blocking read from queue
    bool peek(T& value)
    {
        int64_t index = m_read_index.load(std::memory_order_relaxed);

        // check if queue is empty
        if (index == m_write_index.load(std::memory_order_acquire)) {
            return false;
        }
        value = m_ring_buffer[index & m_buffer_mask];
        return true;
    }

    bool is_empty()
    {
        return m_read_index.load(std::memory_order_acquire) ==
               m_write_index.load(std::memory_order_acquire);
    }

    int guess_size()
    {
        int ret = m_write_index.load(std::memory_order_acquire) -
                  m_read_index.load(std::memory_order_acquire);
        if (ret < 0) {
            ret += m_capacity;
        }
        return ret;
    }

    size_t size()
    {
        return m_ring_buffer.size();
    }

    void shutdown()
    {
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_terminate.store(true, std::memory_order_release);
        }
        m_condition.notify_one();
    }

    void startup()
    {
        m_terminate.store(false);
        m_write_index.store(0);
        m_read_index.store(0);
    }

private:
    std::mutex m_mutex;
    std::condition_variable m_condition;
    std::atomic<bool> m_terminate;
    int64_t m_buffer_mask, m_capacity;
    std::vector<T> m_ring_buffer;
    alignas(64) std::atomic<int64_t> m_write_index;
    alignas(64) std::atomic<int64_t> m_read_index;
    char _pad[64 - sizeof(std::atomic<int64_t>)];
};

#endif // SPSCQUEUE_H
