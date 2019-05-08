#pragma once

#include <vector>
#include <thread>
#include <atomic>
#include "Detector.hh"
#include "drp.hh"

namespace Drp {

class PGPReader
{
public:
    PGPReader(const Parameters& para, MemPool& pool, Detector* det);
    void run();
    void shutdown();
private:
    const Parameters* m_para;
    MemPool* m_pool;
    int32_t dmaRet[MAX_RET_CNT_C];
    uint32_t dmaIndex[MAX_RET_CNT_C];
    uint32_t dest[MAX_RET_CNT_C];
    std::vector<std::thread> m_workerThreads;
    std::atomic<bool> m_terminate;
};

}
