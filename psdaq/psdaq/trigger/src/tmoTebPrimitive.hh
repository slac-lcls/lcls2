/**
 * @file
 * @brief TmoTebPrimitive, the example TMO trigger primitive that writes a TmoTebData record.
 */
#ifndef Pds_Trg_TmoTebPrimitive_hh
#define Pds_Trg_TmoTebPrimitive_hh

#include "TriggerPrimitive.hh"

#include "TmoTebData.hh"

namespace Pds {
  namespace Trg {

    /** Example trigger primitive writing a TmoTebData record with fixed values. */
    class TmoTebPrimitive : public TriggerPrimitive
    {
    public:
      using TriggerPrimitive::configure; // Unhide the Xtc overload, which is not overridden
      /** Does nothing; returns 0. */
      int    configure(const nlohmann::json& configureMsg,
                       const nlohmann::json& connectMsg,
                       size_t                collectionId) override;
      /** Append a TmoTebData with the fixed example values write 0xdeadbeef and monitor 0x12345678 to xtc; the contribution is not inspected. */
      void   event(const Drp::MemPool& pool,
                   uint32_t            idx,
                   const XtcData::Xtc& ctrb,
                   XtcData::Xtc&       xtc,
                   const void*         bufEnd) override;
      /** GPU build (tmoTebPrimitive_gpu.cu): launch a kernel that, in state 1, writes the same fixed values into the TEB input part of the output buffer and moves to state 2. The CPU build (tmoTebPrimitive_cpu.cc) logs a critical error and aborts. */
      void   event(cudaStream_t           stream,
                   unsigned* const        state,
                   float     const* const calibBuffers,
                   size_t    const        calibBufsCnt,
                   uint32_t* const        outBuffers,
                   size_t    const        outBufsCnt,
                   unsigned  const* const index,
                   unsigned* const        retCode_d) override;
      /** Return sizeof(TmoTebData). */
      size_t size() const override  { return sizeof(TmoTebData); }
    };
  }
}

#endif
