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

void Pds::Trg::MfxTripperPrimitive_jungfrau::event(const Drp::MemPool& pool,
                                                   uint32_t            idx,
                                                   const XtcData::Xtc& ctrb,
                                                   XtcData::Xtc&       xtc,
                                                   const void*         bufEnd)
{
    XtcData::ShapesData& shapesData = *reinterpret_cast<XtcData::ShapesData*>(ctrb.payload());
    XtcData::Data& xtcData = shapesData.data();

    const JungfrauData& jungfrauData = *new (xtcData.payload()) JungfrauData;

    new (xtc.alloc(sizeof(TripperTebData), bufEnd)) TripperTebData(jungfrauData.numHotPixels,
                                                                   jungfrauData.maxHotPixels,
                                                                   "jungfrau");
}

// The class factory

extern "C" Pds::Trg::TriggerPrimitive* create_producer_jungfrau()
{
    return new Pds::Trg::MfxTripperPrimitive_jungfrau;
}
