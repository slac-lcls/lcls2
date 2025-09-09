#include "LcReducer.hh"

#include "GpuAsyncLib.hh"


using namespace Drp::Gpu;


LcReducer::LcReducer(const Parameters& para, const MemPoolGpu& pool) :
  Gpu::ReducerAlgo(para, pool, Alg("LC", 0, 0, 0))
{
}

// This routine records the graph that does the data reduction
void LcReducer::recordGraph(cudaStream_t&      stream,
                            const unsigned&    index,
                            float const* const calibBuffers,
                            const size_t       calibBufsCnt,
                            uint8_t    * const dataBuffers,
                            const size_t       dataBufsCnt,
                            unsigned*          extent)
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

extern "C" Drp::Gpu::ReducerAlgo* createReducer(const Drp::Parameters& para, Drp::Gpu::MemPoolGpu& pool)
{
  return new Drp::Gpu::LcReducer(para, pool);
}
