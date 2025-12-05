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
#include "RingIndex_DtoD.hh"

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
  Reader(const Parameters&, MemPoolGpu&, Detector&,
         size_t trgPrimitiveSize, const cuda::atomic<uint8_t>& terminate_d);
  ~Reader();
  void start();
public:
  MemPool& pool()  const { return m_pool; }
  Ptr<RingIndexDtoD>& queue() { return m_readerQueue; }
private:
  int         _setupGraph();
  cudaGraph_t _recordGraph(CUdeviceptr dmaBuffer, CUdeviceptr hwWriteStart);
  void        _reader(Detector&, ReaderMetrics&);
private:
  MemPoolGpu&                  m_pool;
  Ptr<Detector>                m_det;
  const cuda::atomic<uint8_t>& m_terminate_d;
  cudaStream_t                 m_stream;
  cudaGraphExec_t              m_graphExec;
  Ptr<RingIndexDtoD>           m_readerQueue;
  unsigned*                    m_head;
  CUdeviceptr*                 m_dmaBuffers;    // [nFpgas * dmaCount][maxDmaSize]
  CUdeviceptr*                 m_hwWriteStarts; // [nFpgas]
  const Parameters&            m_para;
};

  } // Gpu
} // Drp
