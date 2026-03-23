#include "tmoTebPrimitive.hh"

#include "tmoTebPrimitive_gpu_dev.hh"

#include <cuda_runtime.h>               // For cudaStream_t

using namespace Pds;
using namespace Pds::Trg;
using namespace Drp::Gpu;


static __global__
void _event(float     const* const __restrict__ calibBuffers,
            size_t    const                     calibBufsCnt,
            uint32_t* const* const __restrict__ out,
            size_t    const                     outBufsCnt,
            unsigned  const&                    index,
            size_t    const                     nPanels)
{
  //printf("### TmoTebPrimitive::event: idx %u\n", index);

  TmoTebEventFn{}(calibBuffers, calibBufsCnt, out, outBufsCnt, index, nPanels);

  //printf("### TmoTebPrimitive::event: Done with idx %u\n", index);
}

// This method presumes that it is being called while the stream is in capture mode
void Pds::Trg::TmoTebPrimitive::event(cudaStream_t           stream,
                                      float     const* const calibBuffers,
                                      size_t    const        calibBufsCnt,
                                      uint32_t* const* const out,
                                      size_t    const        outBufsCnt,
                                      unsigned  const&       index,
                                      unsigned  const        nPanels)
{
  printf("*** TmoTebPrimitive::event 1\n");
  _event<<<1, 1, 0, stream>>>(calibBuffers, calibBufsCnt, out, outBufsCnt, index, nPanels);
  printf("*** TmoTebPrimitive::event 2\n");
}

