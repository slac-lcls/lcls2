#include <thread>
#include <atomic>
#include <chrono>
#include <unistd.h>
#include <cstring>
#include "pgpdriver.h"

struct Counters
{
    int64_t total_bytes_received;
    int64_t event_count;
    int64_t pool_size;
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
        printf("Pool size %ld\n", c->pool_size);
        old_bytes = new_bytes;
        old_count = new_count;
    }
}

static void usage(const char* p)
{
    printf("Usage: %s <options>\n", p);
    printf("Options:\n");
    printf("\t-P <bus_id>  [pci bus id of pgg card from lcpsi | grep SLAC]\n");
}


int main(int argc, char* argv[])
{
    char bus_id[32];
    int c;
    while((c = getopt(argc, argv, "P")) != EOF) {
        switch(c) {
            case 'P':  
                strcpy(bus_id, optarg); 
                break;
            default: 
                usage(argv[0]);
                return 0;
        }
    }

    // start monitoring thread
    Counters c1, c2;
    Counters* counter = &c2;
    std::atomic<Counters*> p(&c1);
    std::thread monitor_thread(monitor, std::ref(p));

    int num_entries = 1048576;
    DmaBufferPool pool(num_entries, RX_BUFFER_SIZE);
    AxisG2Device dev;
    dev.init(&pool);       
    dev.setup_lanes(0xF);
    int64_t event_count = 0;
    int64_t total_bytes_received = 0;
    bool validate = true;
    uint32_t event[MAX_LANES];
    while (true) {    
        DmaBuffer* buffer = dev.read();
        if (validate) {
            if (event_count == 0) {
                event[buffer->dest] = ((uint32_t*)buffer->virt)[0];
            } 
            else {
                uint32_t new_event = ((uint32_t*)buffer->virt)[0];
                if ((event[buffer->dest] + 1) != new_event) {
                    printf("Wrong data, expecting, %u, but got %u instead\n", event[buffer->dest]+1, new_event);
                }
                event[buffer->dest] = new_event;
            }
        }
        // printf("Size: %u  Dest: %u\n", buffer->size, buffer->dest);        
        event_count += 1;
        total_bytes_received += buffer->size;
        counter->event_count = event_count;
        counter->total_bytes_received = total_bytes_received;
        counter->pool_size = pool.buffer_queue.guess_size();
        counter = p.exchange(counter, std::memory_order_release);
        // return buffer to memory pool
        pool.buffer_queue.push(buffer);

    }                                

    // shutdown monitor thread
    counter->total_bytes_received = -1;
    p.exchange(counter, std::memory_order_release); 
    monitor_thread.join();    
} 
