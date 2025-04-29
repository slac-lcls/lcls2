#ifndef Pds_Trg_TimingTebData_hh
#define Pds_Trg_TimingTebData_hh

namespace Pds {
  namespace Trg {

    struct TimingTebData
    {
        TimingTebData(const uint8_t   ebeamDestn_,
                      const uint32_t* eventcodes_) {
            ebeamDestn = ebeamDestn_;
            memcpy(eventcodes,eventcodes_,sizeof(eventcodes));
        };
        uint8_t  ebeamDestn;
        uint32_t eventcodes[9];
    };
  };
};

#endif
