#pragma once

#include <vector>
#include <thread>
#include <atomic>
#include "Detector.hh"
#include "drp.hh"
#include "spscqueue.hh"

namespace Pds {
    class MetricExporter;
    namespace Eb { class TebContributor;}
};

namespace Drp {

struct Batch
{
    uint32_t start;
    uint32_t size;
};

class DrpBase;

class PGPDetector : public PgpReader
{
public:
    PGPDetector(const Parameters& para, DrpBase& drp, Detector* det, int* inpMqId, int* resMqId,
                int* inpShmId, int* resShmId, size_t shemeSize);
    ~PGPDetector();
    void reader(std::shared_ptr<Pds::MetricExporter> exporter, Detector* det, Pds::Eb::TebContributor& tebContributor);
    void collector(Pds::Eb::TebContributor& tebContributor);
    void shutdown();
private:
    static const int MAX_RET_CNT_C = 1000;
    std::vector<SPSCQueue<Batch> > m_workerInputQueues;
    std::vector<SPSCQueue<Batch> > m_workerOutputQueues;
    std::vector<std::thread> m_workerThreads;
    std::atomic<bool> m_terminate;
    Batch m_batch;
    unsigned m_nodeId;
    int* m_inpMqId;
    int* m_resMqId;
    int* m_inpShmId;
    int* m_resShmId;
    std::atomic<int> threadCountWrite;
    std::atomic<int> threadCountPush;
    unsigned m_flushTmo;
    size_t m_shmemSize;
};

}
