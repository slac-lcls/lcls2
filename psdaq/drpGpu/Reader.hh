#pragma once

#include "GpuAsyncLib.hh"

#include "GpuAsyncOffsets.h"

#include <cstddef>
#include <vector>
#include <atomic>

#include <cuda_runtime.h>
#include <cuda/atomic>

#include "drp/drp.hh"
#include "MemPool.hh"                   // For Ptr
#include "ringIndex_DtoD.hh"

namespace Pds {
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
};

class Reader
{
public:
  Reader(unsigned panel, const Parameters&, MemPoolGpu&, Detector&,
         size_t trgPrimitiveSize, const cuda::atomic<int>& terminate_d);
  ~Reader();
  void start();
public:
  MemPool& pool()  const { return m_pool; }
  RingIndexDtoD* queue() { return m_readerQueue.d; }
private:
  int         _setupGraphs(unsigned instance);
  cudaGraph_t _recordGraph(unsigned    instance,
                           CUdeviceptr hwWritePtr,
                           CUdeviceptr hwWriteStart);
  void        _reader(Detector&, ReaderMetrics&);
private:
  MemPoolGpu&                  m_pool;
  Detector&                    m_det;
  const cuda::atomic<int>&     m_terminate_d;
  bool*                        m_done;      // Cache for m_terminate_d
  std::vector<cudaStream_t>    m_streams;
  std::vector<cudaGraphExec_t> m_graphExecs;
  Ptr<RingIndexDtoD>           m_readerQueue;
  unsigned*                    m_head[MAX_BUFFERS];
  unsigned                     m_panel;
  const Parameters&            m_para;
};

  } // Gpu
} // Drp
