#ifndef Pds_Trg_TriggerData_cam_hh
#define Pds_Trg_TriggerData_cam_hh

namespace Pds {
  namespace Trg {

    struct TriggerData_cam
    {
      TriggerData_cam(uint64_t value_) : value(value_) {};
      uint64_t value;
    };
  };
};

#endif
