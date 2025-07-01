#include "tmoTebPrimitive.hh"

#include "drpGpu/MemPool.hh"            // For DmaDsc
#include "psdaq/service/EbDgram.hh"     // For TimingHeader

#include <cuda_runtime.h>               // For cudaStream_t

using namespace Pds;
using namespace Pds::Trg;
using namespace Drp::Gpu;

static __global__ void _event(float**    const  __restrict__ calibBuffers,
                              uint32_t** const* __restrict__ out,
                              unsigned&                      index,
                              bool&                          done)
{
  // Analyze calibBuffers for all panels to determine TEB input data for the trigger
  // Example dummy summary input data:
  const uint32_t write_   = 0xdeadbeef;
  const uint32_t monitor_ = 0x12345678;

  unsigned  tebInpOs = (sizeof(DmaDsc) + sizeof(TimingHeader)) / sizeof(***out);
  uint32_t* tebInp   = &out[0][index][tebInpOs];   // Only panel 0 receives the summary
  // @todo: Need a __host__ ctor for TmoTebData?
  //new(tebInp) TmoTebData(write_, monitor_);        // Must be no larger than size()
  tebInp[0] = write_;
  tebInp[1] = monitor_;
}

// This method presumes that it is being called while the stream is in capture mode
void Pds::Trg::TmoTebPrimitive::event(cudaStream_t&     stream,
                                      float**           calibBuffers,
                                      uint32_t** const* out,
                                      unsigned&         index,
                                      bool&             done)
{
  printf("*** TTP:event 1\n");
  _event<<<1, 1, 0, stream>>>(calibBuffers, out, index, done);
  printf("*** TTP:event 2\n");
}
