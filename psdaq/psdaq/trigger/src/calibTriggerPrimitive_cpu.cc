#include "calibTriggerPrimitive.hh"

#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;

// CPU DRPs should not be trying to set up a GPU
void Pds::Trg::CalibPrimitive::event(cudaStream_t           /*stream*/,
                                     unsigned* const        /*state_d*/,
                                     float     const* const /*calibBuffers*/,
                                     size_t    const        /*calibBufsCnt*/,
                                     uint32_t* const        /*outBuffers*/,
                                     size_t    const        /*outBufsCnt*/,
                                     unsigned  const* const /*index*/,
                                     unsigned* const        /*retCode*/)
{
  logging::critical("CalibPrimitive::event for the GPU called by a CPU DRP");
  abort();
}
