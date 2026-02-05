#ifndef Pds_Trg_Trigger_hh
#define Pds_Trg_Trigger_hh

#include "psdaq/service/Dl.hh"
#include "psdaq/eb/eb.hh"               // For MAX_DRPS
#include "psdaq/eb/ResultDgram.hh"
#include "xtcdata/xtc/Dgram.hh"

#include <nlohmann/json.hpp>
#include <cstdint>
#include <string>

namespace Pds {
  namespace Trg {

    class Trigger
    {
    public:
      virtual ~Trigger() {}
    public:
      virtual unsigned rogReserve(unsigned rog,
                                  unsigned meb,
                                  size_t   nBufs) const { return 0; }
      virtual int      configure(const nlohmann::json&      connectMsg,
                                 const nlohmann::json&      configureMsg,
                                 const Pds::Eb::EbParams&   prms) = 0;
      virtual int      initialize(const std::vector<size_t>& inputsRegSizes,
                                  size_t                     resultsRegSize) { return 0; };
      virtual void     event(const Pds::EbDgram* const* start,
                             const Pds::EbDgram**       end,
                             Pds::Eb::ResultDgram&      result) = 0;
      virtual void     transition(Pds::Eb::ResultDgram& result) {}
      virtual void     shutdown() {};
    public:
      static size_t size() { return sizeof(Pds::Eb::ResultDgram); }
    };
  };
};

#endif
