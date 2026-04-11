#pragma once

#include <cstddef>
#include <vector>
#include <atomic>
#include <string>
#include <map>
#include <memory>

#include <cuda_runtime.h>
#include <cuda/std/atomic>

#include "drp/drp.hh"
#include "MemPool.hh"                   // For Ptr
#include "RingIndex_DtoD.hh"

namespace Pds {
  class MetricExporter;
  class TimingHeader;
  namespace Trg {
    class TriggerPrimitive;
  }
}

namespace Drp {
  namespace Gpu {

class Detector;

struct ReaderMetrics
{
  Ptr<uint64_t> state {nullptr, nullptr};
};

class Reader
{
public:
  Reader(const Parameters&, MemPoolGpu&, Detector&, size_t trgPrimitiveSize,
         const cudaExecutionContext_t&, const cuda::std::atomic<unsigned>& terminate_d);
  ~Reader();
  int setupMetrics(const std::shared_ptr<Pds::MetricExporter>,
                   std::map<std::string, std::string>& labels);
  void start();
public:
  MemPool& pool()  const { return m_pool; }
  Ptr<RingIndexDtoD>& queue() { return m_readerQueue; }
private:
  int         _setupGraph();
  cudaGraph_t _recordGraph();
  void        _reader(Detector&, ReaderMetrics&);
private:
  MemPoolGpu&                        m_pool;
  Detector&                          m_det;
  const cudaExecutionContext_t&      m_ctx;
  const cuda::std::atomic<unsigned>& m_terminate_d;
  cudaStream_t                       m_stream;
  cudaGraphExec_t                    m_graphExec;
  Ptr<RingIndexDtoD>                 m_readerQueue;
  unsigned*                          m_head;
  CUdeviceptr*                       m_dmaBuffers;    // [dmaCount][maxDmaSize]
  CUdeviceptr*                       m_fpgaRegs;
  const Parameters&                  m_para;
  ReaderMetrics                      m_metrics;
};

  } // Gpu
} // Drp
