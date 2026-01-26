from libc.stdint cimport uint32_t
cimport psdaq.trigger.EbDgram as dgram

cdef extern from 'psdaq/eb/CubeResultDgram.hh' namespace "Pds::Eb":
    cdef cppclass CubeResultDgram:
        CubeResultDgram(const dgram.EbDgram& dgram, unsigned id) except +
        void     persist(int      value)
        int      persist()
        void     monitor(uint32_t value)
        uint32_t monitor()
        uint32_t data()
        void     bin    (uint32_t value)
        uint32_t bin    ()
        void     worker (uint32_t value)
        uint32_t worker ()
        void     record (int value)
        int      record ()
