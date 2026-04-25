#pragma once

#include <cstddef>
#include <vector>
#include <atomic>

#include <cuda_runtime.h>
#include <cuda/std/atomic>

#include "MemPool.hh"
#include "RingIndex_HtoD.hh"
#include "RingIndex_DtoD.hh"
#include "RingIndex_DtoH.hh"
#include "psdaq/trigger/TriggerPrimitive.hh"
#include "xtcdata/xtc/TransitionId.hh"

namespace Pds {
  class MetricExporter;
  class TimingHeader;
}

namespace Drp {
  namespace Gpu {

class Detector;
class Reader;

struct CollectorMetrics
{
  Ptr<uint64_t> state    {nullptr, nullptr};
  Ptr<uint64_t> rcvWtCtr {nullptr, nullptr};
  Ptr<uint64_t> fwdWtCtr {nullptr, nullptr};

  uint64_t      pndWtCtr     {0};
  uint64_t      pidWtCtr     {0};

  uint64_t      nEvents      {0};
  uint64_t      nDmaRet      {0};
  uint64_t      nHdrMismatch {0};
  uint64_t      dmaSize      {0};
  uint64_t      dmaBytes     {0};
  uint64_t      latency      {0};
  uint64_t      nDmaErrors   {0};
  uint64_t      nNoComRoG    {0};
  uint64_t      nMissingRoGs {0};
  uint64_t      nTmgHdrError {0};
  uint64_t      nPgpJumps    {0};
};

class Collector
{
public:
  Collector(const Parameters&, MemPoolGpu&, const std::shared_ptr<Reader>&,
            Pds::Trg::TriggerPrimitive*, cudaExecutionContext_t,
            const std::atomic<bool>& terminate, const cuda::std::atomic<unsigned>& terminate_d);
  ~Collector(); // = default;
  int setupMetrics(const std::shared_ptr<Pds::MetricExporter>,
                   std::map<std::string, std::string>& labels);
  void start();
  void freeDma(PGPEvent*);
  void handleBrokenEvent(const PGPEvent&) {}
  void resetEventCounter() { m_lastComplete = 0; } // EvtCounter reset
  bool receive(Detector*);
private:
  int _setupGraph();
  cudaGraph_t _recordGraph(cudaStream_t);
  void _freeDma(unsigned index);
private:
  MemPoolGpu&                        m_pool;
  Pds::Trg::TriggerPrimitive*        m_triggerPrimitive;
  Pds::Trg::GpuDispatchType          m_gpuDispatchType;
  const std::atomic<bool>&           m_terminate;
  const cuda::std::atomic<unsigned>& m_terminate_d;
  unsigned*                          m_retCode_d;
  unsigned*                          m_state_d;
  cudaStream_t                       m_stream;
  cudaGraphExec_t                    m_graphExec;
  Ptr<RingIndexHtoD>&                m_pebbleQueue;
  Ptr<RingIndexDtoD>&                m_readerQueue;
  Ptr<RingIndexDtoH>                 m_collectorQueue;
  unsigned*                          m_index_d;
  uint64_t                           m_lastPid;
  uint64_t                           m_latPid;
  uint32_t                           m_lastComplete;
  XtcData::TransitionId::Value       m_lastTid;
  uint32_t                           m_lastData[6];
  const Parameters&                  m_para;
  CollectorMetrics                   m_metrics;
};

  } // Gpu
} // Drp
