#pragma once

#include <cstddef>
#include <vector>
#include <atomic>

#include <cuda_runtime.h>
#include <cuda/std/atomic>

#include "MemPool.hh"
#include "RingIndex_DtoD.hh"
#include "RingIndex_DtoH.hh"
#include "xtcdata/xtc/TransitionId.hh"

namespace Pds {
  class TimingHeader;
  namespace Trg {
    class TriggerPrimitive;
  }
}

namespace Drp {
  namespace Gpu {

class Detector;
class Reader;

struct CollectorMetrics
{
  CollectorMetrics() :
    m_nevents     (0),
    m_nDmaRet     (0),
    m_nHdrMismatch(0),
    m_dmaSize     (0),
    m_dmaBytes    (0),
    m_latency     (0),
    m_nDmaErrors  (0),
    m_nNoComRoG   (0),
    m_nMissingRoGs(0),
    m_nTmgHdrError(0),
    m_nPgpJumps   (0)
  {
  }
  std::atomic<uint64_t> m_nevents;
  std::atomic<uint64_t> m_nDmaRet;
  std::atomic<uint64_t> m_nHdrMismatch;
  std::atomic<uint64_t> m_dmaSize;
  std::atomic<uint64_t> m_dmaBytes;
  std::atomic<uint64_t> m_latency;
  std::atomic<uint64_t> m_nDmaErrors;
  std::atomic<uint64_t> m_nNoComRoG;
  std::atomic<uint64_t> m_nMissingRoGs;
  std::atomic<uint64_t> m_nTmgHdrError;
  std::atomic<uint64_t> m_nPgpJumps;
};

class Collector
{
public:
  Collector(const Parameters&, MemPoolGpu&, const std::shared_ptr<Reader>&, Pds::Trg::TriggerPrimitive*,
            const std::atomic<bool>& terminate, const cuda::std::atomic<unsigned>& terminate_d);
  ~Collector(); // = default;
  void start();
  void freeDma(unsigned index);
  void freeDma(PGPEvent*);
  void handleBrokenEvent(const PGPEvent&) {}
  void resetEventCounter() { m_lastComplete = 0; } // EvtCounter reset
  unsigned receive(Detector*, CollectorMetrics&);
private:
  int         _setupGraph();
  cudaGraph_t _recordGraph(cudaStream_t);
  unsigned    _checkDmaDsc(unsigned index) const;
  unsigned    _checkTimingHeader(unsigned index) const;
private:
  MemPoolGpu&                        m_pool;
  Pds::Trg::TriggerPrimitive*        m_triggerPrimitive;
  const std::atomic<bool>&           m_terminate;
  const cuda::std::atomic<unsigned>& m_terminate_d;
  cudaStream_t                       m_stream;
  cudaGraphExec_t                    m_graphExec;
  Ptr<RingIndexDtoD>&                m_readerQueue;
  Ptr<RingIndexDtoH>                 m_collectorQueue;
  unsigned*                          m_head;
  unsigned*                          m_tail;
  unsigned                           m_last;
  uint64_t                           m_lastPid;
  uint64_t                           m_latPid;
  uint32_t                           m_lastComplete;
  XtcData::TransitionId::Value       m_lastTid;
  uint32_t                           m_lastData[6];
  const Parameters&                  m_para;
};

  } // Gpu
} // Drp
