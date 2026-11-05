#pragma once

#include "ReducerAlgo.hh"

#include <vector>


namespace Drp {
  namespace Gpu {

struct NoOpCounters
{
  unsigned errCnt;
  unsigned blockCount;
};

class NoOpReducer : public ReducerAlgo
{
public:
  NoOpReducer(const Parameters& para, const MemPoolGpu& pool, Detector& det);
  virtual ~NoOpReducer();

  size_t payloadSize() const override { return m_pool.calibBufsSize(); }
  void   recordGraph(cudaStream_t       stream,
                     unsigned*    const state,
                     unsigned*    const index,
                     float const* const calibBuffers,
                     size_t       const calibBufsCnt,
                     uint8_t*     const dataBuffers,
                     size_t       const dataBufsCnt,
                     unsigned*    const retCode) override;
  int      configure(const nlohmann::json& configureMsg,
                     const nlohmann::json& connectMsg,
                     size_t                collectionId) override;
  unsigned configure(XtcData::Xtc&, const void* bufEnd) override;
  void     event    (XtcData::Xtc&, const void* bufEnd, unsigned dataSize) override;
private:
  float const*               m_refBufs_d;
  unsigned                   m_refBufCnt;
  std::vector<NoOpCounters*> m_counters_d;     // One per worker
  unsigned                   m_nGraphs{0};
};

  } // Gpu
} // Drp
