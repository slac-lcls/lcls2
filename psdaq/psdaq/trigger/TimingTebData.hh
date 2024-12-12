#ifndef Pds_Trg_TimingTebData_hh
#define Pds_Trg_TimingTebData_hh

namespace Pds {
  namespace Trg {

    struct TimingTebData
    {
        TimingTebData(const uint32_t* eventcodes_) { memcpy(eventcodes,eventcodes_,sizeof(eventcodes)); };
        uint32_t eventcodes[9];
    };
  };
};

#endif
