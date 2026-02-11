#include "tmoTebPrimitive.hh"

#include "drpGpu/MemPool.hh"            // For DmaDsc
#include "psdaq/service/EbDgram.hh"     // For TimingHeader

#include <cuda_runtime.h>               // For cudaStream_t

using namespace Pds;
using namespace Pds::Trg;
using namespace Drp::Gpu;

static __global__ void _event(float     const* const __restrict__ calibBuffers,
                              const size_t                        calibBufsCnt,
                              uint32_t* const* const __restrict__ out,
                              const size_t                        outBufsCnt,
                              const unsigned&                     index,
                              const size_t                        nPanels)
{
  //printf("### TmoTebPrimitive::event: idx %u\n", index);

  // Analyze calibBuffers for all panels to determine TEB input data for the trigger
  //float* __restrict__ calibBuf = &calibBuffers[index * calibBufsCnt]; // nPanels * nElements of data follow

  // Example dummy summary input data:
  const uint32_t write_  { 0xdeadbeef };
  const uint32_t monitor_{ 0x12345678 };

  // @todo: Maybe out should point directly to where the trigger input data should go?
  constexpr unsigned     tebInpOs = (sizeof(DmaDsc) + sizeof(TimingHeader)) / sizeof(**out);
  uint32_t* __restrict__ tebInp   = &out[0][index * outBufsCnt + tebInpOs];   // Only panel 0 receives the summary
  // @todo: Need a __host__ ctor for TmoTebData?
  //new(tebInp) TmoTebData(write_, monitor_);        // Must be no larger than size()
  tebInp[0] = write_;
  tebInp[1] = monitor_;

  //printf("### TmoTebPrimitive::event: Done with idx %u\n", index);
}

// This method presumes that it is being called while the stream is in capture mode
void Pds::Trg::TmoTebPrimitive::event(cudaStream_t           stream,
                                      float     const* const calibBuffers,
                                      const size_t           calibBufsCnt,
                                      uint32_t* const* const out,
                                      const size_t           outBufsCnt,
                                      const unsigned&        index,
                                      const unsigned         nPanels)
{
  printf("*** TmoTebPrimitive::event 1\n");
  _event<<<1, 1, 0, stream>>>(calibBuffers, calibBufsCnt, out, outBufsCnt, index, nPanels);
  printf("*** TmoTebPrimitive::event 2\n");
}
