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
#include "EventBatcher.hh"              // For EvtBatcherSubFrames
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
  bool setup();
  bool startup();
  void freeDma(PGPEvent*);
  void flush();
public:
  auto& pool()         const { return m_pool; }
  auto& readerQueues() const { return m_readerQueues; }
  auto  nReaders()     const { return m_nReaders; }
  // How the sub-frame scan of the given Reader's last rescanned batch fared.
  // Returns EvtBatcherOk when the Detector's data isn't batched.
  unsigned batcherStatus(unsigned reader) const
  { return m_subFrames[reader].h ? m_subFrames[reader].h->lastStatus() : EvtBatcherOk; }
  // Logs anything the device found wrong with the batch layout.  Returns true on
  // error.  @todo: Provisional.  Whether scan results should reach the host this
  // way or through the per-event hostWrtBufs is still to be settled.
  bool checkBatcher();
private:
  int         _setupGraph(unsigned reader);
  cudaGraph_t _recordGraph(unsigned reader);
private:
  MemPoolGpu&                             m_pool;
  Detector&                               m_det;
  const cudaExecutionContext_t&           m_ctx;
  const cuda::std::atomic<unsigned>&      m_terminate_d;
  std::vector<cudaStream_t>               m_streams;
  std::vector<unsigned*>                  m_dmaBufferIdxes;
  std::vector<unsigned*>                  m_pebbleIdxes;
  std::vector<cudaGraphExec_t>            m_graphExecs;
  Ptr<RingIndexHtoD>                      m_pebbleQueue;
  std::vector< Ptr<RingIndexDtoD> >       m_readerQueues;
  std::vector< Ptr<EvtBatcherSubFrames> > m_subFrames; // Cached scan, per Reader
  std::vector<unsigned*>                  m_states_d;
  unsigned                                m_nReaders;
  const Parameters&                       m_para;
  ReaderMetrics                           m_metrics;
};

  } // Gpu
} // Drp
