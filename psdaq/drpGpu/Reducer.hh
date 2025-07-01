#pragma once

#include <cstddef>
#include <atomic>

#include <cuda_runtime.h>
#include <cuda/atomic>

#include "MemPool.hh"
#include "ringIndex_DtoH.hh"
#include "ringIndex_HtoD.hh"
#include "psdaq/service/Dl.hh"

namespace Drp {
  namespace Gpu {

class Detector;
class ReducerAlgo;

struct ReducerMetrics
{
};

class Reducer
{
public:
  Reducer(unsigned instance, const Parameters&, MemPoolGpu&,
          const std::atomic<bool>& terminate_h, const cuda::atomic<int>& terminate_d);
  ~Reducer();
  void start();
  void stop();
  void reduce(unsigned index);
  //unsigned receive();
  RingIndexDtoH& outputQueue() const { return *m_outputQueue.h; }
private:
  ReducerAlgo* _setupAlgo();
  int          _setupGraph();
  cudaGraph_t  _recordGraph(cudaStream_t&);
private:
  MemPoolGpu&              m_pool;
  Pds::Dl                  m_dl;
  ReducerAlgo*             m_algo;
  const std::atomic<bool>& m_terminate_h;
  const cuda::atomic<int>& m_terminate_d;
  bool*                    m_done;      // Cache for m_terminate_d
  cudaStream_t             m_stream;
  cudaGraph_t              m_graph; // @todo: Goes away?
  cudaGraphExec_t          m_graphExec;
  Ptr<RingIndexHtoD>       m_reducerQueue;
  Ptr<RingIndexDtoH>       m_outputQueue;
  unsigned*                m_head;
  unsigned*                m_tail;
  unsigned                 m_instance;
  const Parameters&        m_para;
};

inline
void Reducer::reduce(unsigned index)
{
  printf("*** Reducer::reduce[%u]: 1 idx %u\n", m_instance, index);
  // Tell GPU to reduce buffer at the given index
  m_reducerQueue.h->produce(index);
  printf("*** Reducer::reduce[%u]: 2\n", m_instance);
}

//inline
//unsigned Reducer::receive()
//{
//  printf("*** Reducer::receive[%u]\n", m_instance);
//  auto idx = m_outputQueue.h->consume();
//  printf("*** Reducer::receive[%u]: idx %u\n", m_instance, idx);
//  return idx;
//}

  } // Gpu
} // Drp
