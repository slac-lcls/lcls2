from libc.stdint cimport uint32_t, uint8_t

cdef extern from 'psdaq/trigger/TimingTebData.hh' namespace "Pds::Trg":
    cdef cppclass TimingTebData:
        TimingTebData(uint8_t  ebeamDestn_,
                      uint32_t* eventcodes_) except +
        uint8_t  ebeamDestn
        uint32_t eventcodes[9]
