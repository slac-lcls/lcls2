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
        void     binIndex      (uint32_t value)
        uint32_t binIndex      ()
        void     updateRecord  (int value)
        int      updateRecord  ()
        void     updateMonitor (int value)
        int      updateMonitor ()
