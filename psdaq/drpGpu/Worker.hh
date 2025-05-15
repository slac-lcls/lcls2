#pragma once

#include "GpuAsyncLib.hh"

#include "GpuAsyncOffsets.h"

#include <cstddef>
#include <vector>
#include <thread>
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

struct WorkerMetrics
{
};

class Worker
{
public:
  Worker(unsigned panel, const Parameters&, MemPoolGpu&, RingIndexDtoD*, Detector& det,
         size_t trgPrimitiveSize, const cuda::atomic<int>& terminate_d);
  ~Worker(); // = default;
  void start();
public:
  MemPool& pool()  const { return m_pool; }
private:
  int     _setupGraphs(int instance);
  CUgraph _recordGraph(cudaStream_t& stream,
                       CUdeviceptr   hwWritePtr,
                       CUdeviceptr   hwWriteStart);
  void    _reader(Detector&, WorkerMetrics&);
private:
  MemPoolGpu&                  m_pool;
  Detector&                    m_det;
  const cuda::atomic<int>&     m_terminate_d;
  bool*                        m_done;      // Cache for m_terminate_d
  std::vector<cudaStream_t>    m_streams;
  std::vector<cudaGraph_t>     m_graphs; // @todo: Goes away?
  std::vector<cudaGraphExec_t> m_graphExecs;
  RingIndexDtoD*               m_workerQueue_d; // Device pointer
  unsigned*                    m_head[MAX_BUFFERS];
  unsigned                     m_panel;
  const Parameters&            m_para;
};

  } // Gpu
} // Drp
