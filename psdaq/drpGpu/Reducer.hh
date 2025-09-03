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
#include "psdaq/service/fast_monotonic_clock.hh"


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
          const std::atomic<bool>&     terminate_h,
          const cuda::atomic<uint8_t>& terminate_d);
  ~Reducer();
  void startup();
  void start(unsigned worker, unsigned index);
  void receive(unsigned worker, unsigned index);
  //void release(unsigned index) const { m_outputQueue.h->release(index); }
  void configure(XtcData::Xtc& xtc, const void* bufEnd)
  { m_algo->configure(xtc, bufEnd); }
  void event(XtcData::Xtc& xtc, const void* bufEnd, size_t dataSize)
  { m_algo->event(xtc, bufEnd, dataSize); }
  uint64_t reduceTime() const { return m_reduce_us; }
private:
  ReducerAlgo* _setupAlgo(Detector&);
  int          _setupGraph(unsigned instance);
  cudaGraph_t  _recordGraph(unsigned instance);
private:
  using timePoint_t = std::chrono::time_point<Pds::fast_monotonic_clock>;
  MemPoolGpu&                   m_pool;
  Pds::Dl                       m_dl;
  ReducerAlgo*                  m_algo;
  const std::atomic<bool>&      m_terminate;
  const cuda::atomic<uint8_t>&  m_terminate_d;
  std::vector<cudaStream_t>     m_streams;
  std::vector<timePoint_t>      m_t0;
  std::vector<cudaGraphExec_t>  m_graphExecs;
  std::vector<unsigned*>        m_heads_h;
  std::vector<unsigned*>        m_heads_d;
  std::vector<unsigned*>        m_tails_h;
  std::vector<unsigned*>        m_tails_d;
  uint64_t                      m_reduce_us;
  const Parameters&             m_para;
};

  } // Gpu
} // Drp
