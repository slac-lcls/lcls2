#pragma once

#include <memory>
#include <thread>
#include <atomic>
#include <nlohmann/json.hpp>
#include "drp/DrpBase.hh"
#include "Worker.hh"
#include "Collector.hh"

class ZmqContext;

namespace Pds {
  class MetricExporter;
  namespace Eb { class TebContributor; }
}

namespace Drp {

class DrpBase;

  namespace Gpu {

class Detector;
class MemPoolGpu;
class RingIndexDtoD;
class RingIndexDtoH;

class PGPDrp : public DrpBase
{
public:
  PGPDrp(Parameters&, MemPoolGpu&, Detector*, ZmqContext&);
  ~PGPDrp();
  std::string configure(const nlohmann::json& msg);
  unsigned unconfigure();
  void collector();
private:
  int _setupMetrics(const std::shared_ptr<Pds::MetricExporter>);
private:
  const Parameters&           m_para;
  Detector*                   m_det;
  std::atomic<bool>           m_terminate_h;    // Avoid PCIe transfer of _d
  cuda::atomic<int>*          m_terminate_d;    // Managed memory pointer
  std::vector<RingIndexDtoD>  m_workerQueues_h; // Preserves lifetime of _d
  RingIndexDtoD*              m_workerQueues_d; // A device pointer to [nPanels]
  std::vector<Worker>         m_workers;        // One worker per panel
  Ptr<RingIndexDtoH>          m_collectorQueue;
  std::unique_ptr<Collector>  m_collector;
  std::thread                 m_collectorThread;
  // @todo: Ptr<RingIndexHtoD>         m_reducerQueue;
  // @todo: std::unique_ptr<Reducer>   m_reducer;
  // @todo: Ptr<RingIndexDtoD>         m_recoderQueue_d;
  // @todo: std::unique_ptr<Recorder>  m_recorder;
  uint64_t                    m_nNoTrDgrams;
  WorkerMetrics               m_wkrMetrics;
  CollectorMetrics            m_colMetrics;
};

  } // Gpu
} // Drp
