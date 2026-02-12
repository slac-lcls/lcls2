#ifndef Pds_Trg_PhaseCavityTebData_hh
#define Pds_Trg_PhaseCavityTebData_hh

namespace Pds {
  namespace Trg {
    struct PhaseCavityTebData
    {
        PhaseCavityTebData(double fitTime1_,
                           double fitTime2_,
                           double charge1_,
                           double charge2_) {
            fitTime1 = fitTime1_;
            fitTime2 = fitTime2_;
            charge1  = charge1_;
            charge2  = charge2_;
        };
        double fitTime1;
        double fitTime2;
        double charge1;
        double charge2;
    };
  };
};

#endif
