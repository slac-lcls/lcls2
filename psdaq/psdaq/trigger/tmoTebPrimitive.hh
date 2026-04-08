#ifndef Pds_Trg_TmoTebPrimitive_hh
#define Pds_Trg_TmoTebPrimitive_hh

#include "TriggerPrimitive.hh"

#include "TmoTebData.hh"

namespace Pds {
  namespace Trg {

    class TmoTebPrimitive : public TriggerPrimitive
    {
    public:
      int    configure(const nlohmann::json& configureMsg,
                       const nlohmann::json& connectMsg,
                       size_t                collectionId) override;
      void   event(const Drp::MemPool& pool,
                   uint32_t            idx,
                   const XtcData::Xtc& ctrb,
                   XtcData::Xtc&       xtc,
                   const void*         bufEnd) override;
      void   event(cudaStream_t           stream,
                   float     const* const calibBuffers,
                   size_t    const        calibBufsCnt,
                   uint32_t* const        out,
                   size_t    const        outBufsCnt,
                   unsigned  const&       index) override;
      GpuDispatchType gpuDispatchType() const override { return GpuDispatchType::TmoTeb; }
    size_t size() const override  { return sizeof(TmoTebData); }
    };
  }
}

#endif
