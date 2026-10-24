/**
 * @file
 * @brief XGmdTebData, the TEB input record for XGMD BLD data.
 */
#ifndef Pds_Trg_XGmdTebData_hh
#define Pds_Trg_XGmdTebData_hh

namespace Pds {
  namespace Trg {
    /** TEB input record for XGMD BLD data (pulse energy, vertical position and severity). */
    struct XGmdTebData
    {
        /** Store the pulse energy, the position and the severity (the float arguments are widened to double). */
        XGmdTebData(float milliJoulesPerPulse_, float posy_, uint64_t severity_) {
            milliJoulesPerPulse = milliJoulesPerPulse_;
            POSY = posy_;
            severity = severity_;
        };
        double milliJoulesPerPulse;  ///< Pulse energy in millijoules, per the field name.
        double POSY;  ///< Vertical position, per the field name.
        uint64_t severity;  ///< Severity word of the BLD record.
    };
  };
};

#endif
