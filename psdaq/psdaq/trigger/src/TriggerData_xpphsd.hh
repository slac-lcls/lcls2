#ifndef Pds_Trg_TriggerData_xpphsd_hh
#define Pds_Trg_TriggerData_xpphsd_hh

namespace Pds {
  namespace Trg {

    struct TriggerData_xpphsd
    {
      TriggerData_xpphsd(uint64_t nPeaks_) : nPeaks(nPeaks_) {};
      uint64_t nPeaks;
    };
  };
};

#endif
