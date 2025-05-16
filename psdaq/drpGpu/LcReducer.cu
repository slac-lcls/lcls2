#include "LcReducer.hh"

#include "GpuAsyncLib.hh"


using namespace Drp::Gpu;


LcReducer::LcReducer(Parameters& para, MemPoolGpu& pool) :
  Gpu::ReducerAlgo(&para, &pool)
{
}

// This routine records the graph that does the data reduction
void LcReducer::recordGraph(cudaStream_t&             stream,
                            const unsigned            index,
                            float* const __restrict__ calibBuffers,
                            float* const __restrict__ dataBuffers)
{
  int* d_fullcarry;
  cudaMalloc((void **)&d_fullcarry, chunks * sizeof(int));
  d_reset<<<1, 1>>>();
  cudaMemset(d_fullcarry, 0, chunks * sizeof(byte));
  d_encode<<<blocks, TPB>>>(dpreencdata, dpreencsize, d_encoded, d_encsize, d_fullcarry);
  cudaFree(d_fullcarry);
  cudaDeviceSynchronize();
}

// The class factory

extern "C" Drp::Gpu::ReducerAlgo* createReducer(Drp::Parameters& para, Drp::Gpu::MemPoolGpu& pool)
{
  return new Drp::Gpu::LcReducer(para, pool);
}
