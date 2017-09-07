#ifndef SPSCQUEUE_H
#define SPSCQUEUE_H

#include <atomic>
#include <vector>
#include <mutex>
#include <condition_variable>

class SPSCQueue {
public:
    SPSCQueue(int capacity);
    void push(uint32_t* value);
    uint32_t* pop();
    bool is_empty();
    int guess_size();
    void shutdown();
private:
    alignas(64) std::atomic<int64_t> write_index;
    alignas(64) std::atomic<int64_t> read_index;
    std::mutex _mutex;
    std::condition_variable _condition;
    std::atomic<bool> terminate;
    int64_t buffer_mask;
    std::vector<uint32_t*> buffer;
};


#endif // SPSCQUEUE_H
