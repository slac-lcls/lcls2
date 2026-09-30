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
#include "RingIndex_HtoD.hh"
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
  std::vector<uint64_t*> states;
  std::vector<uint64_t*> pblWtCtrs;
  std::vector<uint64_t*> dmaWtCtrs;
  std::vector<uint64_t*> fwdWtCtrs;
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
  void freeDma(PGPEvent*);
public:
  auto& pool()         const { return m_pool; }
  auto& readerQueues() const { return m_readerQueues; }
  auto  nReaders()     const { return m_nReaders; }
private:
  int         _setupGraph(unsigned reader);
  cudaGraph_t _recordGraph(unsigned reader);
private:
  MemPoolGpu&                        m_pool;
  Detector&                          m_det;
  const cudaExecutionContext_t&      m_ctx;
  const cuda::std::atomic<unsigned>& m_terminate_d;
  std::vector<cudaStream_t>          m_streams;
  std::vector<unsigned*>             m_dmaBufferIdxes;
  std::vector<unsigned*>             m_pebbleIdxes;
  std::vector<cudaGraphExec_t>       m_graphExecs;
  Ptr<RingIndexHtoD>                 m_pebbleQueue;
  std::vector< Ptr<RingIndexDtoD> >  m_readerQueues;
  std::vector<unsigned*>             m_states_d;
  CUdeviceptr*                       m_dmaBuffers;    // [dmaCount][maxDmaSize]
  CUdeviceptr*                       m_fpgaRegs;
  unsigned                           m_nReaders;
  const Parameters&                  m_para;
  ReaderMetrics                      m_metrics;
};

  } // Gpu
} // Drp
