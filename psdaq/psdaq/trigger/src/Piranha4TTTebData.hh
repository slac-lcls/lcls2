/**
 * @file
 * @brief Piranha4TTTebData, the TEB input record of the Piranha4 timetool results.
 */
#ifndef Pds_Trg_Piranha4TTTebData_hh
#define Pds_Trg_Piranha4TTTebData_hh

namespace Pds {
  namespace Trg {

    /** TEB input record with six timetool values copied from a double array. */
    struct Piranha4TTTebData
    {
        /** Copy payload[0] to payload[5] into the six fields in declaration order; with a null payload only m_ampl is set, to -1. */
        Piranha4TTTebData(double_t* payload) { 
            if (payload) {
                m_ampl        = payload[0];
                m_fltpos      = payload[1];
                m_fltpos_ps   = payload[2];
                m_fltpos_fwhm = payload[3];
                m_amplnxt     = payload[4];
                m_refampl     = payload[5];
            }
            else {
                m_ampl        = -1.;
            }
       };
        double m_ampl;  ///< payload[0]; -1 when there was no payload.
        double m_fltpos;  ///< payload[1].
        double m_fltpos_ps;  ///< payload[2].
        double m_fltpos_fwhm;  ///< payload[3].
        double m_amplnxt;  ///< payload[4].
        double m_refampl;  ///< payload[5].
    };
  };
};

#endif
