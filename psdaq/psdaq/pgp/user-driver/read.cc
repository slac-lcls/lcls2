#include <thread>
#include <atomic>
#include <chrono>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <zmq.h>
#include "pgpdriver.h"

struct Counters
{
    int64_t total_bytes_received;
    int64_t event_count;
    int64_t pool_size;
};

void pin_thread(const pthread_t& th, int cpu)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    int rc = pthread_setaffinity_np(th, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        printf("Error calling pthread_setaffinity_np: %d\n ", rc);
    }
}

void monitor(std::atomic<Counters*>& p)
{
    void* context = zmq_ctx_new();
    void* socket = zmq_socket(context, ZMQ_PUB);
    zmq_connect(socket, "tcp://psdev7b:5559");
    char buffer[2048];

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
        int64_t epoch = std::chrono::duration_cast<std::chrono::duration<int64_t>>(
                        std::chrono::system_clock::now().time_since_epoch()).count();
        int size = snprintf(buffer, 2048, 
                R"({"time": [%ld], "event_rate": [%f], "data_rate": [%f]})", epoch, event_rate, data_rate);
        zmq_send(socket, buffer, size, 0);

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
    pin_thread(pthread_self(), 1);

    // start monitoring thread
    Counters c1, c2;
    Counters* counter = &c2;
    std::atomic<Counters*> p(&c1);
    std::thread monitor_thread(monitor, std::ref(p));

    int num_entries = 1048576;
    DmaBufferPool pool(num_entries, RX_BUFFER_SIZE);
    AxisG2Device dev(0x2032);
    dev.init(&pool);       
    dev.setup_lanes(0xF);
    int64_t event_count = 0;
    int64_t total_bytes_received = 0;
    bool validate = true;
    uint32_t event_counter[MAX_LANES];
    memset(event_counter, 0, sizeof(event_counter));

    while (true) {    
        DmaBuffer* buffer = dev.read();
        if (validate) {
            uint32_t* data = (uint32_t*)buffer->virt;
            uint32_t counter = data[4]&0xffffff;
            uint32_t expected = event_counter[buffer->dest] + 1;
            if (counter != expected) {
                printf("expected %u but got  %u instead\n", expected, counter);
            }
            event_counter[buffer->dest] = counter;
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
