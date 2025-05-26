#pragma once

class cudaStream_t;

namespace Drp {
  struct Parameters;

  namespace Gpu {
    class MemPoolGpu;

class ReducerAlgo
{
public:
  ReducerAlgo(Parameters* para, MemPoolGpu* pool) {}
  virtual ~ReducerAlgo() {}

  virtual void recordGraph(cudaStream_t&             stream,
                           float* const __restrict__ calibBuffer,
                           float* const __restrict__ dataBuffer) = 0;
};

  } // Gpu
} // Drp


extern "C"
{
  typedef Drp::Gpu::ReducerAlgo* reducerAlgoFactoryFn_t(Drp::Parameters&, Drp::Gpu::MemPoolGpu&);

  Drp::Gpu::ReducerAlgo* createReducer(Drp::Parameters& para, Drp::Gpu::MemPoolGpu& pool);
}
