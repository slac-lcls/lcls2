#ifndef Pds_Trg_TmoTebData_hh
#define Pds_Trg_TmoTebData_hh

namespace Pds {
  namespace Trg {

    struct TmoTebData
    {
      TmoTebData(uint32_t write_, uint32_t monitor_) : write(write_), monitor(monitor_) {};
      uint32_t write;
      uint32_t monitor;
    };
  };
};

#endif
