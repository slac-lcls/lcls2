from libc.stdint cimport uint32_t

cdef extern from 'psdaq/trigger/TimingTebData.hh' namespace "Pds::Trg":
    cdef cppclass TimingTebData:
        TimingTebData(uint32_t* eventcodes_) except +
        uint32_t eventcodes[9]
