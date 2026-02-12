#ifndef Pds_Trg_GmdTebData_hh
#define Pds_Trg_GmdTebData_hh

namespace Pds {
  namespace Trg {
    struct GmdTebData
    {
        GmdTebData(const float milliJoulesPerPulse_) {
            milliJoulesPerPulse = milliJoulesPerPulse_;
        };
        double milliJoulesPerPulse;
    };
  };
};

#endif
