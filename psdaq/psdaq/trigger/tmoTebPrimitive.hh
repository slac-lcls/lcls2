#include "TriggerPrimitive.hh"

#include "TmoTebData.hh"

namespace Pds {
  namespace Trg {

    class TmoTebPrimitive : public TriggerPrimitive
    {
    public:
      int    configure(const rapidjson::Document& top,
                       const nlohmann::json&      connectMsg,
                       size_t                     collectionId) override;
      void   event(const Drp::MemPool& pool,
                   uint32_t            idx,
                   const XtcData::Xtc& ctrb,
                   XtcData::Xtc&       xtc,
                   const void*         bufEnd) override;
#ifdef __NVCC__  // Override only if being built for a GPU
      void   event(cudaStream_t&          stream,
                   float     const* const calibBuffers,
                   const size_t           calibBufsCnt,
                   uint32_t* const* const out,
                   const size_t           outBufsCnt,
                   const unsigned&        index,
                   const unsigned         nPanels) override;
#else
      using TriggerPrimitive::event;
#endif
      size_t size() const override  { return sizeof(TmoTebData); }
    };
  }
}
