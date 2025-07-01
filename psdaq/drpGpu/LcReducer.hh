#pragma once

#include "ReducerAlgo.hh"


namespace Drp {
  namespace Gpu {

class LcReducer : public ReducerAlgo
{
public:
  LcReducer(const Parameters& para, MemPoolGpu& pool) {}
  virtual ~LcReducer() {}

  virtual void recordGraph(cudaStream_t&                stream,
                           const unsigned&              index,
                           float** const   __restrict__ calibBuffer,
                           uint8_t** const __restrict__ dataBuffer) override;
};

  } // Gpu
} // Drp
