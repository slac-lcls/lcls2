from libc.stdint cimport int32_t, uint8_t

cdef extern from 'psdaq/trigger/HrEncoderTebData.hh' namespace "Pds::Trg":
    cdef cppclass HrEncoderTebData:
        HrEncoderTebData(uint8_t* payload_) except +
        int32_t m_position
        uint8_t m_encErrCnt
        uint8_t m_missedTrigCnt
        uint8_t m_latches
        uint8_t m_reserved
