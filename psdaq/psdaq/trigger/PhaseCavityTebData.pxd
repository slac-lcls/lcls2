from libc.stdint cimport uint64_t, uint32_t, uint8_t

cdef extern from 'psdaq/trigger/PhaseCavityTebData.hh' namespace "Pds::Trg":
    cdef cppclass PhaseCavityTebData:
        PhaseCavityTebData(float fitTime1_, float fitTime2_, float charge1_, float charge2_, uint64_t severity_) except +
        float fitTime1
        float fitTime2
        float charge1
        float charge2
        uint64_t severity
