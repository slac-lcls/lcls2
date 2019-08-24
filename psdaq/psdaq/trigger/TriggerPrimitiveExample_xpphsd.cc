#include "TriggerPrimitive.hh"
#include "TriggerData_xpphsd.hh"
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

    class TriggerPrimitiveExample_xpphsd : public TriggerPrimitive
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
      static size_t size() { return sizeof(TriggerData_xpphsd); }
    };
  };
};


using namespace Pds::Trg;

int Pds::Trg::TriggerPrimitiveExample_xpphsd::configure(const Document& top,
                                                        const json&     connectMsg,
                                                        size_t          collectionId)
{
  return 0;
}

void Pds::Trg::TriggerPrimitiveExample_xpphsd::event(const Drp::MemPool& pool,
                                                     uint32_t            idx,
                                                     const XtcData::Xtc& ctrb,
                                                     XtcData::Xtc&       xtc)
{
  uint64_t nPeaks = ctrb.sizeofPayload(); // Revisit with a real value

  void* buf  = xtc.alloc(sizeof(TriggerData_xpphsd));
  auto  data = static_cast<TriggerData_xpphsd*>(buf);

  data->nPeaks = nPeaks;
}

// The class factory

extern "C" Pds::Trg::TriggerPrimitive* create_producer_xpphsd()
{
  return new Pds::Trg::TriggerPrimitiveExample_xpphsd;
}
