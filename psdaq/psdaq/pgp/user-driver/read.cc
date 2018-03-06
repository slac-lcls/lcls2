#include <thread>
#include <atomic>
#include <chrono>
#include <unistd.h>
#include "pgpdriver.h"

struct Counters
{
    Counters() : total_bytes_received(0), event_count(0) {}
    int64_t total_bytes_received;
    int64_t event_count;
};

void monitor(std::atomic<Counters*>& p)
{
    Counters* c = p.load(std::memory_order_acquire);
    int64_t old_bytes = c->total_bytes_received;
    int64_t old_count = c->event_count;
    auto t = std::chrono::steady_clock::now();
  
    while(1) {
        sleep(1);
        auto oldt = t;
        t = std::chrono::steady_clock::now();
  
        Counters* c = p.load(std::memory_order_acquire);
        int64_t new_bytes = c->total_bytes_received;
        if (new_bytes == -1) {
            break;
        }
        int64_t new_count = c->event_count;
  
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t - oldt).count();
        double data_rate = double(new_bytes - old_bytes) / duration;
        double event_rate = double(new_count - old_count) / duration * 1.0e3;
        printf("Event rate %.2f kHz    Data rate  %.2f MB/s\n", event_rate, data_rate);
        old_bytes = new_bytes;
        old_count = new_count;
    }
}


int main()
{
    // start monitoring thread
    Counters c1, c2;
    Counters* c = &c2;
    std::atomic<Counters*> p(&c1);
    std::thread monitor_thread(monitor, std::ref(p));

    int num_entries = 1048576;
    Mempool pool(num_entries, RX_BUFFER_SIZE);
    AxisG2Device dev("0000:af:00.0");
    dev.init(&pool);                 
    dev.loop_test(1, 32, 0x41);
    int64_t event_count = 0;
    int64_t total_bytes_received = 0;
    uint32_t event;
    for (size_t i=0; i<60000000; i++) {    
        DmaBuffer* buffer = dev.read();
        if (i == 0) {
            event = ((uint32_t*)buffer->virt)[0];
        } 
        else {
            uint32_t new_event = ((uint32_t*)buffer->virt)[0];
            if ((event + 1) != new_event) {
                printf("Wrong data, expecting, %u, but got %u instead\n", event+1, new_event);
            }
            event = new_event;
        }
        event_count += 1;
        total_bytes_received += buffer->size;
        c->event_count = event_count;
        c->total_bytes_received = total_bytes_received;
        c = p.exchange(c, std::memory_order_release);
        // return buffer to memory pool
        pool.buffer_queue.push(buffer);

    }                                
    dev.loop_test(0, 32, 0x7f0);   

    // shutdown monitor thread
    c->total_bytes_received = -1;
    p.exchange(c, std::memory_order_release); 
    monitor_thread.join();    
} 
