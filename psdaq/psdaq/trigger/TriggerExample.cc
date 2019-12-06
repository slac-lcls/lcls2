#include "Trigger.hh"
#include "TriggerData_cam.hh"
#include "TriggerData_xpphsd.hh"
#include "TriggerData_bld.hh"

#include "utilities.hh"

#include <cstdint>
#include <stdio.h>

using namespace rapidjson;
using json = nlohmann::json;


namespace Pds {
  namespace Trg {

    class TriggerExample : public Trigger
    {
    public:
      int  configure(const json&     connectMsg,
                     const Document& top) override;
      void event(const Pds::EbDgram* const* start,
                 const Pds::EbDgram**       end,
                 Pds::Eb::ResultDgram&          result) override;
    private:
      void  _mapIdToDet(const json&     connectMsg,
                        const Document& top);
    private:
      unsigned _id2Det[Pds::Eb::MAX_DRPS];
    private:
      uint32_t _wrtValue;
      uint32_t _monValue;
      uint32_t _peaksThresh;
      uint32_t _eBeamThresh;
    };
  };
};


void Pds::Trg::TriggerExample::_mapIdToDet(const json&     connectMsg,
                                           const Document& top)
{
  const json& body = connectMsg["body"];

  for (auto it : body["drp"].items())
  {
    unsigned    drpId      = it.value()["drp_id"];
    std::string alias      = it.value()["proc_info"]["alias"];
    size_t      found      = alias.rfind('_');
    std::string detName    = alias.substr(0, found);
    //unsigned    detSegment = std::stoi(alias.substr(found+1, alias.size()));

    if (top.HasMember(detName.c_str()))
      _id2Det[drpId] = top[detName.c_str()].GetUint();
    else
      _id2Det[drpId] = -1;
  }
}

int Pds::Trg::TriggerExample::configure(const json&     connectMsg,
                                        const Document& top)
{
  int      rc = 0;

  _mapIdToDet(connectMsg, top);

# define _FETCH(key, item)                                              \
  if (top.HasMember(key))  item = top[key].GetUint();                   \
  else { fprintf(stderr, "%s:\n  Key '%s' not found\n",                 \
                 __PRETTY_FUNCTION__, key);  rc = -1; }

  // CAM:
  _FETCH("persistValue", _wrtValue);
  _FETCH("monitorValue", _monValue);

  // HSD and XPPHSD:
  _FETCH("peaksThresh",  _peaksThresh);

  // BLD:
  _FETCH("eBeamThresh",  _eBeamThresh);

# undef _FETCH

  return rc;
}

void Pds::Trg::TriggerExample::event(const Pds::EbDgram* const* start,
                                     const Pds::EbDgram**       end,
                                     Pds::Eb::ResultDgram&          result)
{
  const Pds::EbDgram* const* ctrb = start;
  bool                           wrt  = 0;
  bool                           mon  = 0;

  // Accumulate each contribution's input into some sort of overall summary
  do
  {
    switch(_id2Det[(*ctrb)->xtc.src.value()])
    {
      case 0:       // Case value must agree with the value for CAM in ConfigDb
      {
        auto data = reinterpret_cast<struct TriggerData_cam*>((*ctrb)->xtc.payload());

        wrt |= ((data->value      ) & 0x00000000fffffffful) == _wrtValue;
        mon |= ((data->value >> 32) & 0x00000000fffffffful) == _monValue;

        //printf("%s: pid %014lx, input %016lx, wrt %d, mon %d\n",
        //       __PRETTY_FUNCTION__, (*ctrb)->pulseId(), input, wrt, mon);

        break;
      }
      case 1:       // Case value must agree with the value for HSD in ConfigDb
      {
        auto data = reinterpret_cast<struct TriggerData_xpphsd*>((*ctrb)->xtc.payload());

        wrt |= data->nPeaks > _peaksThresh;
        mon |= true;

        //printf("%s: pid %014lx, nPeaks %d, wrt %d, mon %d\n",
        //       __PRETTY_FUNCTION__, (*ctrb)->pulseId(), nPeaks, wrt, mon);

        break;
      }
      case 2:       // Case value must agree with the value for BLD in ConfigDb
      {
        auto data = reinterpret_cast<struct TriggerData_bld*>((*ctrb)->xtc.payload());

        wrt |= data->eBeam > _eBeamThresh;
        mon |= true;

        //printf("%s: pid %014lx, eBeam %ld, thresh %d, wrt %d, mon %d\n",
        //       __PRETTY_FUNCTION__, (*ctrb)->seq.pulseId().value(), data->eBeam, _eBeamThresh, wrt, mon);

        break;
      }
      default:
        break;
    };
  }
  while (++ctrb != end);

  // Insert the trigger values into a Result EbDgram
  unsigned line = 0;                    // Revisit: For future expansion
  result.persist(line, wrt);
  result.monitor(line, mon);
}


// The class factory

extern "C" Pds::Trg::Trigger* create_consumer()
{
  return new Pds::Trg::TriggerExample;
}
