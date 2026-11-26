#include "calibTriggerPrimitive.hh"

#include <cuda_runtime.h>               // For cudaStream_t

using namespace Pds;
using namespace Pds::Trg;


// size() is 0, so nothing is written and the TEB accepts every event.  The state
// advance to 2 is the point: without it TrgInpGen's graph stalls on the first event.
static __global__
void _event(unsigned* const        __restrict__ state,
            unsigned* const        __restrict__ retCode)
{
  if (*state == 1) {
    *retCode = 0;                       // No error
    *state   = 2;                       // Tell _trgInpGenLoop to post the event
  }
}

// This method presumes that it is being called while the stream is in capture mode
int Pds::Trg::CalibPrimitive::event(cudaStream_t           stream,
                                    unsigned* const        state_d,
                                    float     const* const /*calibBuffers*/,
                                    size_t    const        /*calibBufsCnt*/,
                                    uint32_t* const        /*outBuffers*/,
                                    size_t    const        /*outBufsCnt*/,
                                    unsigned  const* const /*index*/,
                                    unsigned* const        retCode_d)
{
  _event<<<1, 1, 0, stream>>>(state_d, retCode_d);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
