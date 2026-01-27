#include "TriggerPrimitive.hh"
#include "TripperTebData.hh"
#include "utilities.hh"
#include "drp/drp.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/Damage.hh"

#include <cstdint>
#include <stdio.h>

#ifdef NDEBUG
#undef NDEBUG                           // To ensure assert() aborts
#endif
#include <cassert>

using json = nlohmann::json;


namespace Pds {
  namespace Trg {

    class MfxTripperPrimitive : public TriggerPrimitive
    {
    public:
        int    configure(const json& configureMsg,
                         const json& connectMsg,
                         size_t      collectionId) override;
        void   event(const Drp::MemPool& pool,
                     uint32_t            idx,
                     const XtcData::Xtc& ctrb,
                     XtcData::Xtc&       xtc,
                     const void*         bufEnd) override;
        // This method can't be left pure virtual for non-GPU use so it is
        // defaulted to an empty block that is never called by non-GPU code
        void   event(cudaStream_t&          stream,
                     float     const* const calibBuffers,
                     const size_t           calibBufsCnt,
                     uint32_t* const* const out,
                     const size_t           outBufsCnt,
                     const unsigned&        index,
                     const unsigned         nPanels) override { assert(false); }
        size_t size() const override { return sizeof(TripperTebData); }
    };
  };
};


using namespace Pds::Trg;

int Pds::Trg::MfxTripperPrimitive::configure(const json& configureMsg,
                                             const json& connectMsg,
                                             size_t      collectionId)
{
    return 0;
}

void Pds::Trg::MfxTripperPrimitive::event(const Drp::MemPool& pool,
                                          uint32_t            idx,
                                          const XtcData::Xtc& ctrb,
                                          XtcData::Xtc&       xtc,
                                          const void*         bufEnd)
{
    new (xtc.alloc(sizeof(TripperTebData), bufEnd)) TripperTebData(0, 0, 0, "NOTRIP");
}

// The class factory

extern "C" Pds::Trg::TriggerPrimitive* create_producer()
{
    return new Pds::Trg::MfxTripperPrimitive;
}
