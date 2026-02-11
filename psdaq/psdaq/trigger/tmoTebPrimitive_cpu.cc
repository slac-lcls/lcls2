#include "tmoTebPrimitive.hh"

#include <stdio.h>

#ifdef NDEBUG
#undef NDEBUG                           // To ensure assert() aborts
#endif
#include <cassert>

// This method can't be left pure virtual for non-GPU use so it is
// defaulted to an empty block that is never called by non-GPU code
void Pds::Trg::TmoTebPrimitive::event(cudaStream_t           stream,
                                      float     const* const calibBuffers,
                                      const size_t           calibBufsCnt,
                                      uint32_t* const* const out,
                                      const size_t           outBufsCnt,
                                      const unsigned&        index,
                                      const unsigned         nPanels)
{
  printf("*** TriggerPrimitive::event\n");
  assert(false);
}
