#pragma once

#include <cstddef>
#include <vector>
#include <thread>
#include <atomic>

#include <cuda_runtime.h>
#include <cuda/std/atomic>

#include "MemPool.hh"
#include "RingQueue_DtoH.hh"
#include "RingQueue_HtoD.hh"
#include "ReducerAlgo.hh"
#include "drp/spscqueue.hh"
#include "psdaq/service/Dl.hh"
#include "psdaq/service/fast_monotonic_clock.hh"


namespace Pds {
  class MetricExporter;
}

namespace Drp {
  namespace Gpu {

class Detector;

struct ReducerMetrics
{
};

    //using ReducerTuple = std::tuple<unsigned, size_t>;
struct ReducerTuple
{
  unsigned index;
  size_t   dataSize;
};

class Reducer
{
public:
  Reducer(const Parameters&, MemPoolGpu&, Detector&,
          const std::atomic<bool>&           terminate_h,
          const cuda::std::atomic<unsigned>& terminate_d);
  ~Reducer();
  int setupMetrics(const std::shared_ptr<Pds::MetricExporter>,
                   std::map<std::string, std::string>& labels);
  void startup();
  void shutdown();
  void start(unsigned worker, unsigned index)
    { if (m_algos[0]->hasGraph()) { m_inputQueues2[worker].h->push(index); }
      else                        { m_inputQueues[worker].push(index); } }
  bool receive(unsigned worker, ReducerTuple& items)
    { return m_algos[0]->hasGraph() ? m_outputQueues2[worker].h->pop(items)
                                    : m_outputQueues[worker].pop(items); }
  void configure(XtcData::Xtc& xtc, const void* bufEnd)
    { if (m_algos.size())  m_algos[0]->configure(xtc, bufEnd); }
  void event(XtcData::Xtc& xtc, const void* bufEnd, size_t dataSize)
    { if (m_algos.size())  m_algos[0]->event(xtc, bufEnd, dataSize); }
private:
  bool        _setupAlgos(Detector&);
  int         _setupGraph(unsigned instance);
  cudaGraph_t _recordGraph(unsigned instance);
  void        _worker(unsigned instance);
private:
  using timePoint_t = std::chrono::time_point<Pds::fast_monotonic_clock>;
  MemPoolGpu&                           m_pool;
  Pds::Dl                               m_dl;
  std::vector<ReducerAlgo*>             m_algos;
  const std::atomic<bool>&              m_terminate;
  const cuda::std::atomic<unsigned>&    m_terminate_d;
  std::vector<SPSCQueue<unsigned> >     m_inputQueues;
  std::vector<SPSCQueue<ReducerTuple> > m_outputQueues;
  std::vector<Ptr<RingQueueHtoD<unsigned> > >     m_inputQueues2;
  std::vector<Ptr<RingQueueDtoH<ReducerTuple> > > m_outputQueues2;
  std::vector<std::thread>              m_threads;
  std::vector<cudaStream_t>             m_streams;
  std::vector<timePoint_t>              m_t0;
  std::vector<cudaGraphExec_t>          m_graphExecs;
  std::vector<unsigned*>                m_heads_h;
  std::vector<unsigned*>                m_heads_d;
  std::vector<unsigned*>                m_tails_h;
  std::vector<unsigned*>                m_tails_d;
  unsigned*                             m_done_d;
  uint64_t                              m_reduce_us;
  const Parameters&                     m_para;
};

  } // Gpu
} // Drp
