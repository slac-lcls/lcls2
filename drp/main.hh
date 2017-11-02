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
    // max of 8 lanes on pgp card
    PGPBuffer buffers[8];
};


