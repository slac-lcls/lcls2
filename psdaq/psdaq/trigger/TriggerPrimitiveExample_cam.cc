#include "TriggerPrimitive.hh"
#include "TriggerData_cam.hh"
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

    class TriggerPrimitiveExample_cam : public TriggerPrimitive
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
      void   event(cudaStream_t           stream,
                   float     const* const calibBuffers,
                   const size_t           calibBufsCnt,
                   uint32_t* const* const out,
                   const size_t           outBufsCnt,
                   const unsigned&        index,
                   const unsigned         nPanels) override { assert(false); }
      size_t size() const override { return sizeof(TriggerData_cam); }
    private:
      unsigned _counter;
      uint32_t _persistValue;
      uint32_t _monitorValue;
    };
  };
};


using namespace Pds::Trg;

int Pds::Trg::TriggerPrimitiveExample_cam::configure(const json& configureMsg,
                                                     const json& connectMsg,
                                                     size_t      collectionId)
{
  int rc = 0;
  const json& top{configureMsg["trigger_body"]};

  _counter = 0;

# define _FETCH(key, item)                                              \
  if (top.find(key) != top.end())  item = top[key];                     \
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
                                                  XtcData::Xtc&       xtc,
                                                  const void*         bufEnd)
{
  uint64_t val = (_counter++ % 2 == 0) ? _persistValue : 0;

  // always monitor every event
  val |= uint64_t(_monitorValue) << 32;

  //printf("%s: counter %08x, val %016lx\n", __PRETTY_FUNCTION__, _counter, val);

  new(xtc.alloc(sizeof(TriggerData_cam), bufEnd)) TriggerData_cam(val);
}

// The class factory

extern "C" Pds::Trg::TriggerPrimitive* create_producer_cam()
{
  return new Pds::Trg::TriggerPrimitiveExample_cam;
}
