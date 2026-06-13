#ifndef Pds_Trg_GmdTebData_hh
#define Pds_Trg_GmdTebData_hh

namespace Pds {
  namespace Trg {
    struct GmdTebData
    {
        GmdTebData(const float milliJoulesPerPulse_,
                   const uint64_t severity_) {
            milliJoulesPerPulse = milliJoulesPerPulse_;
            severity = severity_;
        };
        double milliJoulesPerPulse;
        uint64_t severity;
    };
  };
};

#endif
