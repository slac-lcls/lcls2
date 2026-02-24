from libc.stdint cimport uint64_t, uint32_t, uint8_t

cdef extern from 'psdaq/trigger/XGmdTebData.hh' namespace "Pds::Trg":
    cdef cppclass XGmdTebData:
        XGmdTebData(float milliJoulesPerPulse_, float POSY_, uint64_t severity_) except +
        float milliJoulesPerPulse
        float POSY
        uint64_t severity
