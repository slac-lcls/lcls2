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

// Reader loop timing, kept on the device and logged at Unconfigure (-k readerHist=1).
// The log2 histograms are binned by floor(log2(ns)).
struct ReaderHist
{
  static constexpr unsigned nLog2   {40};
  static constexpr unsigned nBacklog{1024};
  uint64_t tEntry;                      // Last state-0 pass start, %globaltimer ns
  uint64_t tDetect;                     // When the current event's DMA was seen
  uint64_t tPop;                        // When its staging buffer index was obtained
  uint64_t tLoop;                       // When _readerLoop first saw the event done
  uint64_t tPrevDetect;
  unsigned idle;                        // Previous state-0 pass found no DMA
  unsigned nSample;
  uint64_t relaunch[nLog2];             // Period of passes that found no DMA
  uint64_t interval[nLog2];             // Between successive DMA detections
  uint64_t service [nLog2];             // DMA detection to index handed to TrgInpGen
  uint64_t pop     [nLog2];             //   detection to staging buffer obtained
  uint64_t event   [nLog2];             //   that to _readerLoop: _event plus two launches
  uint64_t push    [nLog2];             //   that to the index pushed to TrgInpGen
  uint64_t backlog [nBacklog];          // Later DMA buffers already landed at detection
  uint64_t sum[6];                      // In the order above

  // Failed attempts, each of which costs a graph relaunch
  uint64_t popTimeouts;                 // No staging buffer index available
  uint64_t pushFulls;                   // TrgInpGen's queue full

  // Per-second timeline, to see when a Reader starts falling behind
  static constexpr unsigned sliceShift{30};           // ~1.07 s of %globaltimer
  static constexpr unsigned nSlice    {2048};
  struct Slice
  {
    uint32_t nEvents;                   // Events serviced
    uint32_t popTimeouts;
    uint32_t pushFulls;
    uint32_t maxBacklog;
    uint64_t service;                   // Summed service time, ns
  };
  uint64_t t0;                          // First DMA detected
  unsigned slice;                       // The current event's
  Slice    slices[nSlice];

  // Physical SMs each kernel ran on, as 256-bit masks
  unsigned smWait [8];
  unsigned smEvent[8];
  unsigned smLoop [8];
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
  // The first non-Ok status across all Readers, for a caller that doesn't know
  // which Reader an event came from
  unsigned batcherStatus() const
  {
    for (unsigned i = 0; i < m_nReaders; ++i) {
      auto status = batcherStatus(i);
      if (status != EvtBatcherOk)  return status;
    }
    return EvtBatcherOk;
  }
  // Logs anything the device found wrong with the batch layout.  Returns true on
  // error.  @todo: Provisional.  Whether scan results should reach the host this
  // way or through the per-event hostWrtBufs is still to be settled.
  bool checkBatcher();
  void reportHistograms() const;
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
  std::vector< Ptr<EvtBatcherSubFrames> > m_subFrames;   // Cached scan, per Reader
  std::vector<unsigned*>                  m_states_d;
  std::vector<unsigned*>                  m_evtStatus_d; // Per-event EventStatus
  std::vector<ReaderHist*>                m_hist_d;      // nullptr unless readerHist=1
  unsigned                                m_nReaders;
  const Parameters&                       m_para;
  ReaderMetrics                           m_metrics;
};

  } // Gpu
} // Drp
