/**
 * @file
 * @brief GmdTebData, the TEB input record for GMD BLD data.
 */
#ifndef Pds_Trg_GmdTebData_hh
#define Pds_Trg_GmdTebData_hh

namespace Pds {
  namespace Trg {
    /** TEB input record for GMD BLD data, filled by bldTebPrimitive.cc. */
    struct GmdTebData
    {
        /** Store the pulse energy (widened from float to double) and the severity. */
        GmdTebData(const float milliJoulesPerPulse_,
                   const uint64_t severity_) {
            milliJoulesPerPulse = milliJoulesPerPulse_;
            severity = severity_;
        };
        double milliJoulesPerPulse;  ///< milliJoulesPerPulse value of the GMD BLD record.
        uint64_t severity;  ///< Severity word of the BLD record.
    };
  };
};

#endif
