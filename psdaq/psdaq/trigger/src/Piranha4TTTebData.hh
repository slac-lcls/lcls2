#ifndef Pds_Trg_Piranha4TTTebData_hh
#define Pds_Trg_Piranha4TTTebData_hh

namespace Pds {
  namespace Trg {

    struct Piranha4TTTebData
    {
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
        double m_ampl;
        double m_fltpos;
        double m_fltpos_ps;
        double m_fltpos_fwhm;
        double m_amplnxt;
        double m_refampl;
    };
  };
};

#endif
