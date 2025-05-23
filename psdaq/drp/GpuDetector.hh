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
    int _setupMetrics(const std::shared_ptr<Pds::MetricExporter>);
private:
    const Parameters&       m_para;
    MemPoolGpu&             m_pool;
    Detector*               m_det;
    std::vector<GpuWorker*> m_workers;
    uint64_t                m_nNoTrDgrams;
    GpuMetrics              m_metrics;
};

}
