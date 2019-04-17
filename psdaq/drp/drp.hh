#ifndef DRP_H
#define DRP_H

#include <vector>
#include <cstdint>
#include "spscqueue.hh"
#include "psdaq/eb/eb.hh"

struct DmaBuffer
{
    uint32_t dmaIndex;
    int32_t size;
    void* data;
};

struct PGPData
{
    uint64_t pulse_id;
    uint8_t buffer_mask;
    unsigned damaged : 1;
    unsigned counter : 7;
    DmaBuffer buffers[8];
};

struct Parameters
{
    int partition;
    std::string collect_host;
    std::string output_dir;
    std::string detectorType;
    std::string device;
    int numWorkers;
    int numEntries;
    int laneMask;
    Pds::Eb::TebCtrbParams tPrms;
    Pds::Eb::MebCtrbParams mPrms;
};

// Per-Event-Buffer-with-Boundaries-Listed-Explicitly
class Pebble
{
public:
    Pebble() : _stack(_stack_buffer) {}
    void* fex_data() {return reinterpret_cast<void*>(_fex_buffer);}
    PGPData* pgp_data;
    void* malloc(size_t size)
    {
        void* curr_stack = _stack;
        _stack += size;
        return curr_stack;
    }
private:
    uint8_t _fex_buffer[128*1024];
    uint8_t _stack_buffer[128*1024];
    uint8_t* _stack;
};


using PebbleQueue = SPSCQueue<Pebble*>;

struct MemPool
{
    MemPool(const Parameters& para);
    void** dmaBuffers;
    std::vector<PGPData> pgp_data;
    PebbleQueue pebble_queue;
    std::vector<PebbleQueue> worker_input_queues;
    std::vector<PebbleQueue> worker_output_queues;
    SPSCQueue<int> collector_queue;
    int num_entries;
    // File descriptor for pgp card
    int fd;
// private: FIXME
    std::vector<Pebble> pebble;

};

struct Counters
{
    int64_t total_bytes_received;
    int64_t event_count;
    Counters() : total_bytes_received(0), event_count(0) {}
};

namespace Pds {
    namespace Eb {
        class TebContributor;
        class StatsMonitor;
    };
};

void pin_thread(const pthread_t& th, int cpu);
void monitor_func(const Parameters& para, std::atomic<Counters*>& p,
                  MemPool& pool, Pds::Eb::TebContributor& ebCtrb,
                  Pds::Eb::StatsMonitor& smon);

#endif // DRP_H
