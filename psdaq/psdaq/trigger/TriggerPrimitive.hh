#ifndef Pds_Trg_TriggerPrimitive_hh
#define Pds_Trg_TriggerPrimitive_hh

#include "psdaq/service/Dl.hh"
#include <nlohmann/json.hpp>

#ifdef __NVCC__
#include <cuda_runtime.h>               // For cudaStream_t
#else
struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;
#endif

#include <cstdint>
#include <string>
#include <cassert>

namespace XtcData {
  class Xtc;
}

namespace Drp {
  class MemPool;
}

namespace Pds {
  namespace Trg {

    class TriggerPrimitive
    {
    public:
      virtual ~TriggerPrimitive() {}
    public:
      virtual int    configure(const nlohmann::json& configureMsg,
                               const nlohmann::json& connectMsg,
                               size_t                collectionId) = 0;
      virtual void   configure(const XtcData::Xtc& xtc, const void* bufEnd) {}
      virtual void   event(const Drp::MemPool& pool,
                           uint32_t            index,
                           const XtcData::Xtc& contribution,
                           XtcData::Xtc&       xtc,
                           const void*         bufEnd) = 0;
      // This method can't be left pure virtual for non-GPU use so it is
      // defaulted to an empty block that is never called by non-GPU code
      virtual void   event(cudaStream_t&          stream,
                           float     const* const calibBuffers,
                           const size_t           calibBufsCnt,
                           uint32_t* const* const out,
                           const size_t           outBufsCnt,
                           const unsigned&        index,
                           const unsigned         nPanels) { assert(false); } // = 0;
      virtual size_t size() const = 0;
    };
  }
}

#endif
