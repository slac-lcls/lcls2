/**
 * @file
 * @brief GasDetTebData, the TEB input record for gas detector BLD data.
 */
#ifndef Pds_Trg_GasDetTebData_hh
#define Pds_Trg_GasDetTebData_hh

namespace Pds {
  namespace Trg {
    /** TEB input record holding the six gas detector values named as in BldNames::GasDet. */
    struct GasDetTebData
    {
        /** Store the six values. */
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
        double f11ENRC;  ///< Value f11ENRC of the gas detector BLD record.
        double f12ENRC;  ///< Value f12ENRC of the gas detector BLD record.
        double f21ENRC;  ///< Value f21ENRC of the gas detector BLD record.
        double f22ENRC;  ///< Value f22ENRC of the gas detector BLD record.
        double f63ENRC;  ///< Value f63ENRC of the gas detector BLD record.
        double f64ENRC;  ///< Value f64ENRC of the gas detector BLD record.
    };
  };
};

#endif
