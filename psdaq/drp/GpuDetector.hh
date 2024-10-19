#pragma once

#include <memory>
#include <thread>
#include <atomic>
#include "drp.hh"
#include "spscqueue.hh"
#include "Detector.hh"
#include "DrpBase.hh"

namespace Pds {
    class MetricExporter;
    namespace Eb { class TebContributor; }
};

namespace Drp {

struct Batch
{
    uint32_t start;
    uint32_t size;
};

class GpuWorker;

class GpuDetector : public PgpReader
{
public:
    GpuDetector(const Parameters& para, DrpBase& drp, GpuWorker* gpu);
    virtual ~GpuDetector();
    void reader(std::shared_ptr<Pds::MetricExporter> exporter, Detector* det, Pds::Eb::TebContributor& tebContributor);
    void collector(Pds::Eb::TebContributor& tebContributor);
    virtual void handleBrokenEvent(const PGPEvent& event) override;
    virtual void resetEventCounter() override;
    void shutdown();
private:
    void _gpuCollector(Pds::Eb::TebContributor& tebContributor);
private:
    static constexpr int MAX_RET_CNT_C { 8 };
    DrpBase&             m_drp;
    GpuWorker*           m_gpu;
    Detector*            m_det;
    SPSCQueue<uint32_t>  m_collectorCpuQueue;
    SPSCQueue<Batch>     m_collectorGpuQueue;
    std::atomic<bool>    m_terminate;
};

}
