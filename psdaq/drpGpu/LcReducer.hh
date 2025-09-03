#pragma once

#include "ReducerAlgo.hh"


namespace Drp {
  namespace Gpu {

class LcReducer : public ReducerAlgo
{
public:
  LcReducer(const Parameters& para, MemPoolGpu& pool) {}
  virtual ~LcReducer() {}

  virtual void recordGraph(cudaStream_t&      stream,
                           const unsigned&    index,
                           float const* const calibBuffers,
                           const size_t       calibBufsCnt,
                           uint8_t    * const dataBuffers,
                           const size_t       dataBufsCnt,
                           unsigned*          extent) override;
};

  } // Gpu
} // Drp
