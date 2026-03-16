from libc.stdint cimport uint32_t
cimport psdaq.trigger.EbDgram as dgram

cdef extern from 'psdaq/eb/CubeConfigDgram.hh' namespace "Pds::Eb":
    cdef cppclass CubeConfigDgram:
        CubeConfigDgram(const dgram.EbDgram& dgram, unsigned id) except +
        void     bins          (uint32_t value)
        void     appendJson    (char* value)
