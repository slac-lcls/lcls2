#ifndef Pds_Trg_GasDetTebData_hh
#define Pds_Trg_GasDetTebData_hh

namespace Pds {
  namespace Trg {
    struct GasDetTebData
    {
        GasDetTebData(double f11ENRC_,
                      double f12ENRC_,
                      double f21ENRC_,
                      double f22ENRC_,
                      double f63ENRC_,
                      double f64ENRC_) {
            f11ENRC = f11ENRC_;
            f12ENRC = f12ENRC_;
            f21ENRC = f21ENRC_;
            f22ENRC = f22ENRC_;
            f63ENRC = f63ENRC_;
            f64ENRC = f64ENRC_;
        };
        double f11ENRC;
        double f12ENRC;
        double f21ENRC;
        double f22ENRC;
        double f63ENRC;
        double f64ENRC;
    };
  };
};

#endif
