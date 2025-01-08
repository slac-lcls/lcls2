#pragma once

#include <memory>
#include <thread>
#include <atomic>
#include "drp.hh"
#include "spscqueue.hh"
#include "GpuWorker.hh"

namespace Pds {
    class MetricExporter;
    namespace Eb { class TebContributor; }
};

namespace Drp {

class DrpBase;
class Detector;
class MemPoolGpu;

class GpuDetector
{
public:
    GpuDetector(const Parameters& para, MemPoolGpu& pool, Detector* det);
    ~GpuDetector();
    void collector(std::shared_ptr<Pds::MetricExporter>, Pds::Eb::TebContributor&, DrpBase&);
    void shutdown();
private:
    const Parameters&               m_para;
    MemPoolGpu&                     m_pool;
    Detector*                       m_det;
    std::vector<GpuWorker*>         m_workers;
    std::vector< SPSCQueue<Batch> > m_workerQueues;
    std::atomic<bool>               m_terminate;
    uint64_t                        m_nNoTrDgrams;
};

}
