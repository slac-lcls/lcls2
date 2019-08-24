#include "TriggerPrimitive.hh"
#include "TriggerData_bld.hh"
#include "utilities.hh"
#include "drp/drp.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/Damage.hh"

#include <cstdint>
#include <stdio.h>

using namespace rapidjson;
using json = nlohmann::json;


namespace Pds {
  namespace Trg {

    class TriggerPrimitiveExample_bld : public TriggerPrimitive
    {
    public:
      int    configure(const Document& top,
                       const json&     connectMsg,
                       size_t          collectionId) override;
      void   event(const Drp::MemPool& pool,
                   uint32_t            idx,
                   const XtcData::Xtc& ctrb,
                   XtcData::Xtc&       xtc) override;
    public:
      static size_t size() { return sizeof(TriggerData_bld); }
    };
  };
};


using namespace Pds::Trg;

int Pds::Trg::TriggerPrimitiveExample_bld::configure(const Document& top,
                                                     const json&     connectMsg,
                                                     size_t          collectionId)
{
  return 0;
}

void Pds::Trg::TriggerPrimitiveExample_bld::event(const Drp::MemPool& pool,
                                                  uint32_t            idx,
                                                  const XtcData::Xtc& ctrb,
                                                  XtcData::Xtc&       xtc)
{
  uint32_t* bld   = reinterpret_cast<uint32_t*>(ctrb.payload());
  uint64_t  eBeam = ctrb.damage.value() ? 0 : bld[2]; // Revisit: do something real

  void* buf  = xtc.alloc(sizeof(TriggerData_bld));
  auto  data = static_cast<TriggerData_bld*>(buf);

  data->eBeam = eBeam;
}

// The class factory

extern "C" Pds::Trg::TriggerPrimitive* create_producer_bld()
{
  return new Pds::Trg::TriggerPrimitiveExample_bld;
}
