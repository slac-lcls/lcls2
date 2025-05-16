#pragma once

#include "ReducerAlgo.hh"


namespace Drp {
  namespace Gpu {

class NoOpReducer : public ReducerAlgo
{
public:
  NoOpReducer(Parameters* para, MemPoolGpu* pool) {}
  virtual ~NoOpReducer() {}

  virtual void recordGraph(cudaStream_t&             stream,
                           float* const __restrict__ calibBuffer,
                           float* const __restrict__ dataBuffer) override;
};

  } // Gpu
} // Drp
