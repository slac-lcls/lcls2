from libc.stdint cimport uint64_t, uint32_t, uint8_t

cdef extern from 'psdaq/trigger/GmdTebData.hh' namespace "Pds::Trg":
    cdef cppclass GmdTebData:
        GmdTebData(float milliJoulesPerPulse_) except +
        float milliJoulesPerPulse
