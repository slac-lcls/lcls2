#ifndef SPSCQUEUE_H
#define SPSCQUEUE_H

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>

template<typename T>
class SPSCQueue
{
public:
    SPSCQueue(int capacity) : write_index(0), read_index(0), terminate(false)
    {
        buffer.resize(capacity);
        buffer_mask = capacity - 1;
    }
    void push(T value)
    {
        int64_t index = write_index.load();
        buffer[index & buffer_mask] = value;
        int64_t next = index + 1;
        write_index.store(next); // std::memory_order_release
        // signal consumer
        if (index == read_index.load(std::memory_order_acquire)) {
            std::unique_lock<std::mutex> lock(_mutex);
            _condition.notify_one();
        }
    }
    bool pop(T& value)
    {
        int64_t index = read_index.load();

        // Queue is empty
        if (index == write_index.load()) { // std::memory_order_acquire
            std::unique_lock<std::mutex> lock(_mutex);
            _condition.wait(lock, [this] { return !is_empty() || terminate.load(); });
            if (terminate && is_empty()) {
                return false;
            }
        }

        value = buffer[index & buffer_mask];
        int64_t next = index + 1;
        read_index.store(next, std::memory_order_release);
        return true;
    }
    bool is_empty()
    {
        return read_index.load() == write_index.load();
    }
    int guess_size()
    {
        int ret = write_index.load(std::memory_order_acquire) - read_index.load(std::memory_order_acquire);
        return std::max(0, ret);
    }
    void shutdown()
    {
        {
            std::unique_lock<std::mutex> lock(_mutex);
            terminate.store(true);
        }
        _condition.notify_one();
    }

private:
    std::mutex _mutex;
    char _pad1[64 - sizeof(std::mutex)];

    std::atomic<int64_t> write_index;
    char _pad2[64 - sizeof(std::atomic<int64_t>)];

    std::atomic<int64_t> read_index;
    char _pad3[64 - sizeof(std::atomic<int64_t>)];

    std::condition_variable _condition;
    std::atomic<bool> terminate;
    int64_t buffer_mask;
    std::vector<T> buffer;
};

#endif // SPSCQUEUE_H
