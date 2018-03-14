#ifndef SPSCQUEUE_H
#define SPSCQUEUE_H

#include <atomic>
#include <cassert>
#include <mutex>
#include <vector>
#include <condition_variable>

template <typename T>
class SPSCQueue
{
public:
    SPSCQueue(int capacity) : m_terminate(false), m_write_index(0), m_read_index(0)
    {
        assert((capacity & (capacity - 1)) == 0);
        m_ring_buffer.resize(capacity);
        m_capacity = capacity;
        m_buffer_mask = capacity - 1;
    }

    SPSCQueue(const SPSCQueue&) = delete;
    void operator=(const SPSCQueue&) = delete;

    SPSCQueue(SPSCQueue&& d)
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
    bool pop(T& value)
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

    void shutdown()
    {
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_terminate.store(true, std::memory_order_release);
        }
        m_condition.notify_one();
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
