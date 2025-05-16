#pragma once

#include "ReducerAlgo.hh"


namespace Drp {
  namespace Gpu {

class LcReducer : public ReducerAlgo
{
public:
  LcReducer(Parameters* para, MemPoolGpu* pool) {}
  virtual ~LcReducer() {}

  virtual void recordGraph(cudaStream_t&             stream,
                           float* const __restrict__ calibBuffer,
                           float* const __restrict__ dataBuffer) override;
};

  } // Gpu
} // Drp
