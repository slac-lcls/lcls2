#ifndef Pds_Trg_XGmdTebData_hh
#define Pds_Trg_XGmdTebData_hh

namespace Pds {
  namespace Trg {
    struct XGmdTebData
    {
        XGmdTebData(float milliJoulesPerPulse_, float posy_, uint64_t severity_) {
            milliJoulesPerPulse = milliJoulesPerPulse_;
            POSY = posy_;
            severity = severity_;
        };
        double milliJoulesPerPulse;
        double POSY;
        uint64_t severity;
    };
  };
};

#endif
