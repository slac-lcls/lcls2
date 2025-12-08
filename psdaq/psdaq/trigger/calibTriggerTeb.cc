#include "Trigger.hh"

#include "utilities.hh"

#include <cstdint>
#include <stdio.h>

using json = nlohmann::json;


namespace Pds {
  namespace Trg {

    class CalibTrigger : public Trigger
    {
    public:
      int  configure(const json&              connectMsg,
                     const json&              configureMsg,
                     const Pds::Eb::EbParams& prms) override;
      void event(const Pds::EbDgram* const* start,
                 const Pds::EbDgram**       end,
                 Pds::Eb::ResultDgram&      result) override;
    };
  };
};


int Pds::Trg::CalibTrigger::configure(const json&              connectMsg,
                                      const json&              configureMsg,
                                      const Pds::Eb::EbParams& prms)
{
  return 0;                             // Nothing to do
}

void Pds::Trg::CalibTrigger::event(const Pds::EbDgram* const* start,
                                   const Pds::EbDgram**       end,
                                   Pds::Eb::ResultDgram&      result)
{
  // Insert the trigger values into a Result EbDgram, accepting all data
  result.persist(true);
  result.monitor(-1);                   // All MEBs
}


// The class factory

extern "C" Pds::Trg::Trigger* create_consumer()
{
  return new Pds::Trg::CalibTrigger;
}
