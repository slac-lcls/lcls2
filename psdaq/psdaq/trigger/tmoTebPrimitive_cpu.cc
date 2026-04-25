#include "tmoTebPrimitive.hh"

#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;

// CPU DRPs should not be trying to set up a GPU
void Pds::Trg::TmoTebPrimitive::event(cudaStream_t           /*stream*/,
                                      unsigned* const        /*state_d*/,
                                      float     const* const /*calibBuffers*/,
                                      size_t    const        /*calibBufsCnt*/,
                                      uint32_t* const        /*outBuffers*/,
                                      size_t    const        /*outBufsCnt*/,
                                      unsigned  const* const /*index*/,
                                      unsigned* const        /*retCode*/)
{
  logging::critical("TmoTebPrimitive::setupGpu called by a CPU DRP");
  abort();
}
