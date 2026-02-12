from libc.stdint cimport uint64_t, uint32_t, uint8_t

cdef extern from 'psdaq/trigger/EBeamTebData.hh' namespace "Pds::Trg":
    cdef cppclass EBeamTebData:
        EBeamTebData(float l3Energy_) except +
        float l3Energy
