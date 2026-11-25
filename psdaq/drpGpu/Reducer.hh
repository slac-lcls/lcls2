#pragma once

#include <cstddef>
#include <vector>
#include <atomic>
#include <string>
#include <map>
#include <memory>

#include <cuda_runtime.h>
#include <cuda/std/atomic>

#include "MemPool.hh"
#include "RingQueue_DtoH.hh"
#include "RingQueue_HtoD.hh"
#include "ReducerAlgo.hh"
#include "StageHist.hh"
#include "psdaq/service/Dl.hh"
#include <nlohmann/json.hpp>
#include "psdaq/service/fast_monotonic_clock.hh"


namespace Pds {
  class MetricExporter;
}

namespace XtcData {
  class Xtc;
}

namespace Drp {
  namespace Gpu {

class Detector;

struct ReducerMetrics
{
  std::vector<uint64_t*> state;
  std::vector<uint64_t*> inpWtCtr;
  std::vector<uint64_t*> outWtCtr;
};

class Reducer
{
public:
  Reducer(const Parameters&, MemPoolGpu&, Detector&,
          cudaExecutionContext_t             green_ctx,
          const std::atomic<bool>&           terminate_h,
          const cuda::std::atomic<unsigned>& terminate_d);
  ~Reducer();
  int setupMetrics(const std::shared_ptr<Pds::MetricExporter>,
                   std::map<std::string, std::string>& labels);
  int configure(const nlohmann::json& configureMsg,
                const nlohmann::json& connectMsg,
                size_t                collectionId);
  bool setup(XtcData::Xtc& xtc, const void* be);
  bool startup();
  void dump() const;
  void reportHistograms() const;
  bool start(unsigned worker, unsigned index)
    { return m_inputQueues[worker].h->push(index); }
  bool receive(unsigned worker, ReducerTuple* items)
    { return m_outputQueues[worker].h->pop(items); }
  void event(XtcData::Xtc& xtc, const void* bufEnd, size_t dataSize)
    { if (m_algo)  m_algo->event(xtc, bufEnd, dataSize); }
private:
  bool        _setupAlgo(Detector&);
  int         _setupGraph(unsigned instance);
  cudaGraph_t _recordGraph(unsigned instance);
private:
  MemPoolGpu&                                     m_pool;
  Pds::Dl                                         m_dl;
  ReducerAlgo*                                    m_algo;
  std::atomic<bool> const&                        m_terminate;
  cuda::std::atomic<unsigned> const&              m_terminate_d;
  std::vector<unsigned*>                          m_retCode_d;
  std::vector<unsigned*>                          m_state_d;
  std::vector<Ptr<RingQueueHtoD<unsigned> > >     m_inputQueues;
  std::vector<Ptr<RingQueueDtoH<ReducerTuple> > > m_outputQueues;
  std::vector<cudaStream_t>                       m_streams;
  std::vector<cudaGraphExec_t>                    m_graphExecs;
  std::vector<unsigned*>                          m_indices_d;
  std::vector<StageHist*>                         m_hist_d;      // nullptrs unless readerHist=1
  std::vector<StageTime*>                         m_time_d;
  uint64_t                                        m_timePrev[2];
  Parameters const&                               m_para;
  ReducerMetrics                                  m_metrics;
};

  } // Gpu
} // Drp
