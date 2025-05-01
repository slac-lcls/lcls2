#include "TriggerPrimitive.hh"
#include "TripperTebData.hh"
#include "drp/drp.hh"
#include "utilities.hh"
#include "xtcdata/xtc/Damage.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/ConfigIter.hh"

#include <cstdint>
#include <stdio.h>
#include <iostream>

using namespace rapidjson;
using json = nlohmann::json;


namespace Pds {
  namespace Trg {

    class MfxTripperPrimitive_jungfrau : public TriggerPrimitive
    {
    public:
        int    configure(const Document& top,
                         const json&     connectMsg,
                         size_t          collectionId) override;
        void   event(const Drp::MemPool& pool,
                     uint32_t            idx,
                     const XtcData::Xtc& ctrb,
                     XtcData::Xtc&       xtc,
                     const void*         bufEnd) override;
        void   event(cudaStream_t&     stream,
                     float*            calibBuffers,
                     uint32_t** const* out,
                     unsigned&         index,
                     bool&             done) override;
        size_t size() const  { return sizeof(TripperTebData); }
    };
  };
};


using namespace Pds::Trg;

int Pds::Trg::MfxTripperPrimitive_jungfrau::configure(const Document& top,
                                                      const json&     connectMsg,
                                                      size_t          collectionId)
{
    return 0;
}

#pragma pack(push, 1)
struct JungfrauData {
  uint16_t raw[512*1024];
  uint64_t frame_cnt;
  uint64_t timestamp;
  uint16_t hotPixelThresh;
  uint32_t numHotPixels;
  uint32_t maxHotPixels;
};
#pragma pack(pop)


/** The Jungfrau drp process may contain multiple modules which appear as individual
 *  XTCs per datagram. For providing a TEB contribution, the number of hot pixels in each
 *  module is summed and only a single contribution for ALL segments/drp is sent to the TEB.
 *  The values of `hotPixelThresh` and `maxHotPixels`, while in theory can be different
 *  between modules, should in practice always be identical so only a single value is sent
 *  to the TEB for the time being.
 */
void Pds::Trg::MfxTripperPrimitive_jungfrau::event(const Drp::MemPool& pool,
                                                   uint32_t            idx,
                                                   const XtcData::Xtc& ctrb,
                                                   XtcData::Xtc&       xtc,
                                                   const void*         bufEnd)
{
    int remaining = ctrb.sizeofPayload();
    uint16_t hotPixelThresh{0};
    uint32_t numHotPixels{0};
    uint32_t maxHotPixels{0};

    XtcData::Xtc* payload = reinterpret_cast<XtcData::Xtc*>(ctrb.payload());
    while (remaining > 0) {
        XtcData::ShapesData& shapesData = *reinterpret_cast<XtcData::ShapesData*>(payload);
        XtcData::Data& xtcData = shapesData.data();

        const JungfrauData& jungfrauData = *new (xtcData.payload()) JungfrauData;
        hotPixelThresh = jungfrauData.hotPixelThresh;
        numHotPixels += jungfrauData.numHotPixels; // Sum hot pixels for each module
        maxHotPixels = jungfrauData.maxHotPixels;
        remaining -= payload->sizeofPayload() + sizeof(XtcData::Xtc);
        payload = payload->next();
    }
    new (xtc.alloc(sizeof(TripperTebData), bufEnd)) TripperTebData(hotPixelThresh,
                                                                   numHotPixels,
                                                                   maxHotPixels,
                                                                   "jungfrau");
}

void Pds::Trg::MfxTripperPrimitive_jungfrau::event(cudaStream_t&     stream,
                                                   float*            calibBuffers,
                                                   uint32_t** const* out,
                                                   unsigned&         index,
                                                   bool&             done)
{
  assert(false);  // Unused but must be defined
}

// The class factory

extern "C" Pds::Trg::TriggerPrimitive* create_producer_jungfrau()
{
    return new Pds::Trg::MfxTripperPrimitive_jungfrau;
}
