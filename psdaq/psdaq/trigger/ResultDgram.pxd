from libc.stdint cimport uint32_t
cimport psdaq.trigger.EbDgram as dgram

cdef extern from 'psdaq/eb/ResultDgram.hh' namespace "Pds::Eb":
    cdef cppclass ResultDgram:
        ResultDgram(const dgram.EbDgram& dgram, unsigned id) except +
        void     persist(int      value)
        int      persist()
        void     monitor(uint32_t value)
        uint32_t monitor()
        void     auxdata(uint32_t value)
        uint32_t auxdata()
        uint32_t data()