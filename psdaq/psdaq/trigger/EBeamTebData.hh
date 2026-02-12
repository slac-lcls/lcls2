#ifndef Pds_Trg_EBeamTebData_hh
#define Pds_Trg_EBeamTebData_hh

namespace Pds {
  namespace Trg {
    struct EBeamTebData
    {
        EBeamTebData(double l3Energy_) {
            l3Energy = l3Energy_;
        };
        double l3Energy;
    };
  };
};

#endif
