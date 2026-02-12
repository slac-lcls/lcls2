#ifndef Pds_Trg_XGmdTebData_hh
#define Pds_Trg_XGmdTebData_hh

namespace Pds {
  namespace Trg {
    struct XGmdTebData
    {
        XGmdTebData(float milliJoulesPerPulse_, float posy_) {
            milliJoulesPerPulse = milliJoulesPerPulse_;
            POSY = posy_;
        };
        double milliJoulesPerPulse;
        double POSY;
    };
  };
};

#endif
