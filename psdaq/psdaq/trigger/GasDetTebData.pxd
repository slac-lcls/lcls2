from libc.stdint cimport uint64_t, uint32_t, uint8_t

cdef extern from 'psdaq/trigger/GasDetTebData.hh' namespace "Pds::Trg":
    cdef cppclass GasDetTebData:
        GasDetTebData(float f11ENRC_,
                      float f12ENRC_,
                      float f21ENRC_,
                      float f22ENRC_,
                      float f63ENRC_,
                      float f64ENRC_) except +
        float f11ENRC
        float f12ENRC
        float f21ENRC
        float f22ENRC
        float f63ENRC
        float f64ENRC
