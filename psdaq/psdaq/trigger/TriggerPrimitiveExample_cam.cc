#include "TriggerPrimitive.hh"
#include "TriggerData_cam.hh"
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

    class TriggerPrimitiveExample_cam : public TriggerPrimitive
    {
    public:
      int    configure(const Document& top,
                       const json&     connectMsg,
                       size_t          collectionId) override;
      void   event(const Drp::MemPool& pool,
                   uint32_t            idx,
                   const XtcData::Xtc& ctrb,
                   XtcData::Xtc&       xtc) override;
      size_t size() const  { return sizeof(TriggerData_cam); }
    private:
      unsigned _counter;
      uint32_t _persistValue;
      uint32_t _monitorValue;
    };
  };
};


using namespace Pds::Trg;

int Pds::Trg::TriggerPrimitiveExample_cam::configure(const Document& top,
                                                     const json&     connectMsg,
                                                     size_t          collectionId)
{
  int rc = 0;

  _counter = 0;

# define _FETCH(key, item)                                              \
  if (top.HasMember(key))  item = top[key].GetUint();                   \
  else { fprintf(stderr, "%s:\n  Key '%s' not found\n",                 \
                 __PRETTY_FUNCTION__, key);  rc = -1; }

  _FETCH("persistValue", _persistValue);
  _FETCH("monitorValue", _monitorValue);

# undef _FETCH

  return rc;
}

void Pds::Trg::TriggerPrimitiveExample_cam::event(const Drp::MemPool& pool,
                                                  uint32_t            idx,
                                                  const XtcData::Xtc& ctrb,
                                                  XtcData::Xtc&       xtc)
{
  uint64_t val = (_counter++ % 2 == 0) ? _persistValue : 0;

  // always monitor every event
  val |= uint64_t(_monitorValue) << 32;

  //printf("%s: counter %08x, val %016lx\n", __PRETTY_FUNCTION__, _counter, val);

  void* buf  = xtc.alloc(sizeof(TriggerData_cam));
  auto  data = static_cast<TriggerData_cam*>(buf);
  data->value = val;
}

// The class factory

extern "C" Pds::Trg::TriggerPrimitive* create_producer_cam()
{
  return new Pds::Trg::TriggerPrimitiveExample_cam;
}
