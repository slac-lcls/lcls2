#ifndef Pds_Trg_TriggerPrimitive_hh
#define Pds_Trg_TriggerPrimitive_hh

#include "psdaq/service/Dl.hh"
#include <nlohmann/json.hpp>

#include "rapidjson/document.h"

#ifdef __NVCC__
#include <cuda_runtime.h>               // For cudaStream_t
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
      virtual int    configure(const rapidjson::Document& top,
                               const nlohmann::json&      connectMsg,
                               size_t                     collectionId) = 0;
      virtual void   event(const Drp::MemPool& pool,
                           uint32_t            index,
                           const XtcData::Xtc& contribution,
                           XtcData::Xtc&       xtc,
                           const void*         bufEnd) = 0;
#ifdef __NVCC__
      // This method is for GPU use only and shouldn't show up in CPU builds
      virtual void   event(cudaStream_t&     stream,
                           float*            calibBuffers,
                           uint32_t** const* out,
                           unsigned&         index,
                           bool&             done) = 0;
#endif
      virtual size_t size() const = 0;
    };
  }
}

#endif
