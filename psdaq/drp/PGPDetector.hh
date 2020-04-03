#pragma once

#include <vector>
#include <thread>
#include <atomic>
#include "Detector.hh"
#include "drp.hh"
#include "spscqueue.hh"

class MetricExporter;
namespace Pds {namespace Eb { class TebContributor;}};

namespace Drp {

struct Batch
{
    uint32_t start;
    uint32_t size;
};

class DrpBase;

class PGPDetector
{
public:
    PGPDetector(const Parameters& para, DrpBase& drp, Detector* det);
    void reader(std::shared_ptr<MetricExporter> exporter, Detector* det, Pds::Eb::TebContributor& tebContributor);
    void collector(Pds::Eb::TebContributor& tebContributor);
    void resetEventCounter();
    void shutdown();
private:
    const Parameters& m_para;
    MemPool& m_pool;
    static const int MAX_RET_CNT_C = 1000;
    int32_t dmaRet[MAX_RET_CNT_C];
    uint32_t dmaIndex[MAX_RET_CNT_C];
    uint32_t dest[MAX_RET_CNT_C];
    std::vector<SPSCQueue<Batch> > m_workerInputQueues;
    std::vector<SPSCQueue<Batch> > m_workerOutputQueues;
    std::vector<std::thread> m_workerThreads;
    std::atomic<bool> m_terminate;
    Batch m_batch;
    uint32_t m_lastComplete;
    unsigned m_nodeId;
};

}
