#ifndef PGPREADER_H
#define PGPREADER_H

#include <vector>
#include <thread>
#include "drp.hh"

#define MAX_RET_CNT_C 100

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

class Detector;

class PGPReader
{
public:
    PGPReader(MemPool& pool, Detector* det, int lane_mask, int nworkers);
    PGPData* process_lane(uint32_t lane, uint32_t index, int32_t size);
    void send_to_worker(Pebble* pebble_data);
    void send_all_workers(Pebble* pebble);
    void run();
    std::atomic<Counters*>& get_counters() {return m_pcounter;};
private:
    MemPool& m_pool;
    int m_nlanes;
    int m_buffer_mask;
    uint32_t m_last_complete;
    uint64_t m_worker;
    int m_nworkers;
    MovingAverage m_avg_queue_size;
    Counters m_c1, m_c2;
    std::atomic<Counters*> m_pcounter;
    uint32_t m_dmaIndex[MAX_RET_CNT_C];
    uint32_t m_dmaDest[MAX_RET_CNT_C];
    int32_t m_dmaRet[MAX_RET_CNT_C];
    std::vector<std::thread> m_workerThreads;
};

#endif // PGPREADER_H
