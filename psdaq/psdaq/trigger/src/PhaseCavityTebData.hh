/**
 * @file
 * @brief PhaseCavityTebData, the TEB input record for phase cavity BLD data.
 */
#ifndef Pds_Trg_PhaseCavityTebData_hh
#define Pds_Trg_PhaseCavityTebData_hh

namespace Pds {
  namespace Trg {
    /** TEB input record holding the phase cavity values named as in BldNames::PCav, plus a severity word. */
    struct PhaseCavityTebData
    {
        /** Store the four values and the severity. */
        PhaseCavityTebData(double fitTime1_,
                           double fitTime2_,
                           double charge1_,
                           double charge2_,
                           uint64_t severity_) {
            fitTime1 = fitTime1_;
            fitTime2 = fitTime2_;
            charge1  = charge1_;
            charge2  = charge2_;
            severity = severity_;
        };
        double fitTime1;  ///< Value fitTime1 of the phase cavity BLD record.
        double fitTime2;  ///< Value fitTime2 of the phase cavity BLD record.
        double charge1;  ///< Value charge1 of the phase cavity BLD record.
        double charge2;  ///< Value charge2 of the phase cavity BLD record.
        uint64_t severity;  ///< Severity word of the BLD record.
    };
  };
};

#endif
