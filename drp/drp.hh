#ifndef DRP_H
#define DRP_H

#include <vector>
#include <cstdint>

struct EventHeader {
    uint64_t pulseId;
    uint64_t timeStamp;
    uint32_t l1Count;
    uint32_t version;
    unsigned rawSamples:24;
    unsigned channelMask:8;
    uint32_t reserved;
};

class MovingAverage
{
public:
    MovingAverage(int n);
    int add_value(int value);
private:
    int64_t index;
    int sum;
    int N;
    std::vector<int> values;
};

struct PGPBuffer
{
    uint32_t length;
    uint32_t dma_index;
};

struct PGPData
{
    uint64_t pulse_id;
    uint8_t lane_mask;
    unsigned damaged : 1;
    unsigned counter : 7;
    // max of 8 lanes on pgp card
    PGPBuffer buffers[8];
};

// Per-Event-Buffer-with-Boundaries-Listed-Explicitly
class Pebble
{
public:
    Pebble():_stack(_stack_buffer){}
    void* fex_data() {return reinterpret_cast<void*>(_fex_buffer);}
    PGPData* pgp_data;
    void *malloc(size_t size) {
        void *curr_stack = _stack;
        _stack += size;
        return curr_stack;
    }
private:
    uint8_t _fex_buffer[1024*1024];
    uint8_t _stack_buffer[1024*1024];
    uint8_t *_stack;
};

#endif // DRP_H
