from libc.stdint cimport uint32_t
cimport psdaq.trigger.EbDgramw as dgram

cdef extern from 'psdaq/eb/src/WindowResultDgram.hh' namespace "Pds::Eb":
    cdef cppclass WindowResultDgram:
        WindowResultDgram(const dgram.EbDgram& dgram, unsigned id) except +
        void     persist(int      value)
        int      persist()
        void     monitor(uint32_t value)
        uint32_t monitor()
        uint32_t data()
        void     updateAdd     (uint32_t value)
        uint32_t updateAdd     ()
        void     updateRecord  (uint32_t value)
        uint32_t updateRecord  ()
        void     updateMonitor (uint32_t value)
        uint32_t updateMonitor ()
        void     flush         (uint32_t value)
        uint32_t flush         ()
