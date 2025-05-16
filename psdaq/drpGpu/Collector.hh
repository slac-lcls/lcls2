#pragma once

#include <cstddef>
#include <vector>
#include <thread>
#include <atomic>

#include <cuda_runtime.h>
#include <cuda/atomic>

#include "MemPool.hh"
#include "ringIndex_DtoD.hh"
#include "ringIndex_DtoH.hh"

namespace Pds {
  class TimingHeader;
  namespace Trg {
    class TriggerPrimitive;
  }
}

namespace Drp {
  namespace Gpu {

class Detector;

struct CollectorMetrics
{
  CollectorMetrics() :
    m_nevents     (0),
    m_nDmaRet     (0),
    m_dmaBytes    (0),
    m_dmaSize     (0),
    m_nDmaErrors  (0),
    m_nNoComRoG   (0),
    m_nMissingRoGs(0),
    m_nTmgHdrError(0),
    m_nPgpJumps   (0)
  {
  }
  std::atomic<uint64_t> m_nevents;
  std::atomic<uint64_t> m_nDmaRet;
  std::atomic<uint64_t> m_dmaBytes;
  std::atomic<uint64_t> m_dmaSize;
  std::atomic<uint64_t> m_nDmaErrors;
  std::atomic<uint64_t> m_nNoComRoG;
  std::atomic<uint64_t> m_nMissingRoGs;
  std::atomic<uint64_t> m_nTmgHdrError;
  std::atomic<uint64_t> m_nPgpJumps;
};

class Collector
{
public:
  Collector(const Parameters&, MemPoolGpu&, RingIndexDtoD*, const Ptr<RingIndexDtoH>&, Pds::Trg::TriggerPrimitive*,
            const std::atomic<bool>& terminate_h, const cuda::atomic<int>& terminate_d);
  ~Collector(); // = default;
  void start();
  void freeDma(unsigned index);
  void freeDma(PGPEvent*);
  void handleBrokenEvent(const PGPEvent&);
  void resetEventCounter();
  unsigned receive(Detector*, CollectorMetrics&);
private:
  int     _setupGraph();
  CUgraph _recordGraph(cudaStream_t& stream);
  unsigned _checkDmaDsc(unsigned index) const;
  unsigned _checkTimingHeader(unsigned index) const;
private:
  MemPoolGpu&                 m_pool;
  Pds::Trg::TriggerPrimitive* m_triggerPrimitive;
  const std::atomic<bool>&    m_terminate_h;
  const cuda::atomic<int>&    m_terminate_d;
  bool*                       m_done;      // Cache for m_terminate_d
  cudaStream_t                m_stream;
  cudaGraph_t                 m_graph; // @todo: Goes away?
  cudaGraphExec_t             m_graphExec;
  RingIndexDtoD*              m_workerQueues_d; // A device pointer to [nPanels]
  const Ptr<RingIndexDtoH>&   m_collectorQueue;
  unsigned*                   m_head;
  unsigned*                   m_tail;
  unsigned                    m_last;
  uint64_t                    m_lastPid;
  const Parameters&           m_para;
};

  } // Gpu
} // Drp
