#ifndef Pds_Trg_TriggerData_bld_hh
#define Pds_Trg_TriggerData_bld_hh

namespace Pds {
  namespace Trg {

    struct TriggerData_bld
    {
      TriggerData_bld(uint64_t eBeam_) : eBeam(eBeam_) {};
      uint64_t eBeam;
    };
  };
};

#endif
