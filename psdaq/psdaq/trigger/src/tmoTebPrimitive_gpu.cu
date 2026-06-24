#include "tmoTebPrimitive.hh"

#include "drpGpu/MemPool.hh"            // For DmaDsc
#include "psdaq/service/EbDgram.hh"     // For TimingHeader

#include <cuda_runtime.h>               // For cudaStream_t

using namespace Pds;
using namespace Pds::Trg;
using namespace Drp::Gpu;


static __global__
void _event(float     const* const __restrict__ /*calibBuffers*/,
            size_t    const                     /*calibBufsCnt*/,
            uint32_t* const        __restrict__ outBuffers,
            size_t    const                     outBufsCnt,
            unsigned* const        __restrict__ state,
            unsigned  const* const __restrict__ index,
            unsigned* const        __restrict__ retCode)
{
  // Check that we're in the expected state
  if (*state == 1) {
    //printf("### TmoTebPrimitive::event: idx %u\n", *index);

    // Analyze calibBuffers for all panels to determine TEB input data for the trigger
    //float* __restrict__ calibBuf = &calibBuffers[*index * calibBufsCnt]; // nElements of data follow

    constexpr uint32_t write_{0xdeadbeef};
    constexpr uint32_t monitor_{0x12345678};

    // Although this is also executed for transitions, it is harmless, but it could be tested for here
    constexpr unsigned tebInpOs = (sizeof(DmaDsc) + sizeof(TimingHeader)) / sizeof(*outBuffers);
    uint32_t* const    tebInp   = &outBuffers[*index * outBufsCnt + tebInpOs];
    tebInp[0] = write_;
    tebInp[1] = monitor_;

    *retCode = 0;                         // No error
    //printf("### TmoTebPrimitive::event: Done with idx %u\n", *index);

    // Advance to the next state
    *state = 2;
  }
}

// This method presumes that it is being called while the stream is in capture mode
void Pds::Trg::TmoTebPrimitive::event(cudaStream_t           stream,
                                      unsigned* const        state_d,
                                      float     const* const calibBuffers,
                                      size_t    const        calibBufsCnt,
                                      uint32_t* const        outBuffers,
                                      size_t    const        outBufsCnt,
                                      unsigned  const* const index_d,
                                      unsigned* const        retCode_d)
{
  printf("*** TmoTebPrimitive::event 1\n");
  _event<<<1, 1, 0, stream>>>(calibBuffers,
                              calibBufsCnt,
                              outBuffers,
                              outBufsCnt,
                              state_d,
                              index_d,
                              retCode_d);
  chkError(cudaGetLastError(), "Launch of TmoTebPrimitive _event kernel failed");
  printf("*** TmoTebPrimitive::event 2\n");
}

