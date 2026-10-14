#include "calibTriggerPrimitive.hh"

#include <cuda_runtime.h>               // For cudaStream_t

using namespace Pds;
using namespace Pds::Trg;


// No TEB input data is produced: size() is 0, so there is nowhere to write and the
// TEB accepts every event regardless.  The state advance is the whole purpose of this
// kernel -- TrgInpGen's graph is a three-stage state machine in which this is the
// middle stage, and _trgInpGenLoop only posts an event once state reaches 2.  Omitting
// it stalls the graph on the first event.
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
void Pds::Trg::CalibPrimitive::event(cudaStream_t           stream,
                                     unsigned* const        state_d,
                                     float     const* const /*calibBuffers*/,
                                     size_t    const        /*calibBufsCnt*/,
                                     uint32_t* const        /*outBuffers*/,
                                     size_t    const        /*outBufsCnt*/,
                                     unsigned  const* const /*index*/,
                                     unsigned* const        retCode_d)
{
  _event<<<1, 1, 0, stream>>>(state_d, retCode_d);
}
