#pragma once

#include <cstddef>
#include <atomic>

#include <cuda_runtime.h>
#include <cuda/atomic>

#include "MemPool.hh"
#include "RingIndex_DtoH.hh"
#include "RingIndex_HtoD.hh"
#include "ReducerAlgo.hh"
#include "psdaq/service/Dl.hh"

namespace Drp {
  namespace Gpu {

class Detector;

struct ReducerMetrics
{
};

class Reducer
{
public:
  Reducer(const Parameters&, MemPoolGpu&, Detector&,
          const std::atomic<bool>& terminate_h,
          const cuda::atomic<int>& terminate_d);
  ~Reducer();
  void start(unsigned worker, unsigned index);
  unsigned receive(unsigned worker, unsigned index);
  void release(unsigned index) const { m_outputQueue.h->release(index); }
  void configure(XtcData::Xtc& xtc, const void* bufEnd)
  { m_algo->configure(xtc, bufEnd); }
  void event(XtcData::Xtc& xtc, const void* bufEnd, size_t dataSize)
  { m_algo->event(xtc, bufEnd, dataSize); }
private:
  ReducerAlgo* _setupAlgo(Detector&);
  int          _setupGraph(unsigned instance);
  cudaGraph_t  _recordGraph(unsigned instance);
private:
  MemPoolGpu&                  m_pool;
  Pds::Dl                      m_dl;
  ReducerAlgo*                 m_algo;
  const std::atomic<bool>&     m_terminate_h;
  const cuda::atomic<int>&     m_terminate_d;
  bool*                        m_done;  // Cache for m_terminate_d
  std::vector<cudaStream_t>    m_streams;
  std::vector<cudaGraphExec_t> m_graphExecs;
  Ptr<RingIndexHtoD>           m_reducerQueue;
  Ptr<RingIndexDtoH>           m_outputQueue;
  std::vector<unsigned*>       m_heads;
  std::vector<unsigned*>       m_tails;
  const Parameters&            m_para;
};

  } // Gpu
} // Drp
