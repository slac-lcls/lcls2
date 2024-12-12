#ifndef Pds_Trg_TriggerData_timing_hh
#define Pds_Trg_TriggerData_timing_hh

namespace Pds {
  namespace Trg {

    struct TriggerData_timing
    {
        TriggerData_timing(uint32_t* eventcodes_) { memcpy(eventcodes,eventcodes_,sizeof(eventcodes)); };
        uint32_t eventcodes[9];
    };
  };
};

#endif
