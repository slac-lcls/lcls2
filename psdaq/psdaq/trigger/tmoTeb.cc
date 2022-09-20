#include "Trigger.hh"
#include "TmoTebData.hh"

#include "utilities.hh"

#include <cstdint>
#include <stdio.h>

using namespace rapidjson;
using json = nlohmann::json;


namespace Pds {
  namespace Trg {

    class TmoTeb : public Trigger
    {
    public:
      int  configure(const json&              connectMsg,
                     const Document&          top,
                     const Pds::Eb::EbParams& prms) override;
      void event(const Pds::EbDgram* const* start,
                 const Pds::EbDgram**       end,
                 Pds::Eb::ResultDgram&      result) override;
    private:
      uint32_t _wrtValue;
      uint32_t _monValue;
    };
  };
};


int Pds::Trg::TmoTeb::configure(const json& connectMsg,
                                const Document& top,
                                const Pds::Eb::EbParams& prms)
{
  int rc = 0;

# define _FETCH(key, item)                                              \
  if (top.HasMember(key))  item = top[key].GetUint();                   \
  else { fprintf(stderr, "%s:\n  Key '%s' not found\n",                 \
                 __PRETTY_FUNCTION__, key);  rc = -1; }

  _FETCH("persistValue", _wrtValue);
  _FETCH("monitorValue", _monValue);

# undef _FETCH

  return rc;
}

void Pds::Trg::TmoTeb::event(const Pds::EbDgram* const* start,
                             const Pds::EbDgram**       end,
                             Pds::Eb::ResultDgram&      result)
{
  const Pds::EbDgram* const* ctrb = start;
  bool                       wrt  = false;
  bool                       mon  = false;

  // Accumulate each contribution's input into some sort of overall summary
  do
  {
    auto data = reinterpret_cast<struct TmoTebData*>((*ctrb)->xtc.payload());

    wrt |= data->write   == _wrtValue;
    mon |= data->monitor == _monValue;

    //printf("%s: pid %014lx, input %016lx, wrt %d, mon %d\n",
    //       __PRETTY_FUNCTION__, (*ctrb)->pulseId(), input, wrt, mon);
  }
  while (++ctrb != end);

  // Insert the trigger values into a Result EbDgram
  result.persist(wrt);
  result.monitor(mon ? -1 : 0);         // All MEBs
}


// The class factory

extern "C" Pds::Trg::Trigger* create_consumer()
{
  return new Pds::Trg::TmoTeb;
}
