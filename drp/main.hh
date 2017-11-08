#include <vector>
#include <cstdint>

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
    uint32_t nlanes;
    uint64_t pulse_id;
    uint8_t lane_count;
    uint8_t lane_mask;
    // max of 8 lanes on pgp card
    PGPBuffer buffers[8];
};

// Per-Event-Buffer-with-Boundaries-Listed-Explicitly
class Pebble
{
public:
    PGPData* pgp_data() {return reinterpret_cast<PGPData*>(_pebble_buffer);}
    void* fex_data() {return reinterpret_cast<void*>(_pebble_buffer + sizeof(PGPData));} 
private:
    uint8_t _pebble_buffer[1024*1024];
};
