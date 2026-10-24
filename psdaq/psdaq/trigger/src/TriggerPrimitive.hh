/**
 * @file
 * @brief TriggerPrimitive, the interface of DRP plugins that compute the TEB input data of each event.
 */
#ifndef Pds_Trg_TriggerPrimitive_hh
#define Pds_Trg_TriggerPrimitive_hh

#include "psdaq/service/Dl.hh"
#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>

struct CUstream_st;
/** Forward declaration of the CUDA stream handle type, so that this header does not need the CUDA headers. */
typedef struct CUstream_st* cudaStream_t;

namespace XtcData {
  class Xtc;
}

namespace Drp {
  class MemPool;
}

namespace Pds {
  namespace Trg {

    /** Interface of the DRP plugins that produce the TEB input data (size() bytes) for each event, either on the CPU or in a CUDA stream. */
    class TriggerPrimitive
    {
    public:
      /** Does nothing; empty virtual destructor. */
      virtual ~TriggerPrimitive() {}
    public:
      /** Pure virtual: configure from the configure and connect messages; implementations return 0 on success. */
      virtual int    configure(const nlohmann::json& configureMsg,
                               const nlohmann::json& connectMsg,
                               size_t                collectionId) = 0;
      /** Hook to add to the Configure data in xtc; does nothing unless overridden. */
      virtual void   configure(const XtcData::Xtc& xtc, const void* bufEnd) {}
      /** Pure virtual: append the TEB input data for the event contribution (pebble buffer index of pool) to xtc. */
      virtual void   event(const Drp::MemPool& pool,
                           uint32_t            index,
                           const XtcData::Xtc& contribution,
                           XtcData::Xtc&       xtc,
                           const void*         bufEnd) = 0;
      /** GPU hook: enqueue on stream the work that writes the TEB input data; does nothing unless overridden. */
      virtual void   event(cudaStream_t           stream,
                           unsigned* const        state,
                           float     const* const calibBuffers,
                           size_t    const        calibBufsCnt,
                           uint32_t* const        outBuffers,
                           size_t    const        outBufsCnt,
                           unsigned  const* const index,
                           unsigned* const        retCode_d) {}
      /** Pure virtual: return the size in bytes of the TEB input data. */
      virtual size_t size() const = 0;
    };
  }
}

#endif
