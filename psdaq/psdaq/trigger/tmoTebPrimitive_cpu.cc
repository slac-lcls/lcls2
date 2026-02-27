#include "tmoTebPrimitive.hh"

#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;

// CPU DRPs should not be trying to set up a GPU
void Pds::Trg::TmoTebPrimitive::event(cudaStream_t           stream,
                                      float     const* const calibBuffers,
                                      size_t    const        calibBufsCnt,
                                      uint32_t* const* const out,
                                      size_t    const        outBufsCnt,
                                      unsigned  const&       index,
                                      unsigned  const        nPanels)
{
  logging::critical("TriggerPrimitive::setupGpu called by a CPU DRP");
  abort();
}
