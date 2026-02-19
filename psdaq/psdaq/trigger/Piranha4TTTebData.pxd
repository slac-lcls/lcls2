from libc.stdint cimport int32_t, uint8_t

cdef extern from 'psdaq/trigger/Piranha4TTTebData.hh' namespace "Pds::Trg":
    cdef cppclass Piranha4TTTebData:
        Piranha4TTTebData(float* payload_) except +
        float m_ampl
        float m_fltpos
        float m_fltpos_ps
        float m_fltpos_fwhm
        float m_amplnxt
        float m_refampl

