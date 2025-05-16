#include "NoOpReducer.hh"

#include "GpuAsyncLib.hh"


using namespace Drp::Gpu;


NoOpReducer::NoOpReducer(Parameters& para, MemPoolGpu& pool) :
  Gpu::ReducerAlgo(&para, &pool)
{
}

// This routine records the graph that does the data reduction
void NoOpReducer::recordGraph(cudaStream_t&             stream,
                              const unsigned            index,
                              float* const __restrict__ calibBuffers,
                              float* const __restrict__ dataBuffers)
{
  const auto               panelSize   = NumAsics*NumRows*NumCols;
  const auto& __restrict__ calibBuffer = calibBuffers[index];
  auto&       __restrict__ dataBuffer  = dataBuffers[index];
  chkError(cudaMemcpy(dataBuffer, calibBuffer, panelSize * sizeof(float), cudaMemcpyDeviceToDevice));
}

// The class factory

extern "C" Drp::Gpu::ReducerAlgo* createReducer(Drp::Parameters& para, Drp::Gpu::MemPoolGpu& pool)
{
  return new Drp::Gpu::NoOpReducer(para, pool);
}
