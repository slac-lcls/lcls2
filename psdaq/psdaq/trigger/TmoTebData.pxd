from libc.stdint cimport uint32_t

cdef extern from 'psdaq/trigger/TmoTebData.hh' namespace "Pds::Trg":
    cdef cppclass TmoTebData:
        TmoTebData(uint32_t write_, uint32_t monitor_) except +
        uint32_t write
        uint32_t monitor
