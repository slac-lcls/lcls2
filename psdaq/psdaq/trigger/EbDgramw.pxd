from libc.stdint cimport uint16_t, uint32_t, uint64_t


cdef extern from 'xtcdata/xtc/Src.hh' namespace "XtcData":
    cdef cppclass Src:
        Src() except +
        unsigned value() const

cdef extern from 'xtcdata/xtc/Damage.hh' namespace "XtcData":
    cdef cppclass Damage:
        Damage() except +
        uint16_t value() const

cdef extern from 'xtcdata/xtc/TypeId.hh' namespace "XtcData":
    cdef cppclass TypeId:
        TypeId() except +
        unsigned value() const;

cdef extern from 'xtcdata/xtc/Xtc.hh' namespace "XtcData":
    cdef cppclass Xtc:
        Xtc() except +
        char* payload() const
        int sizeofPayload() const
        Src      src
        Damage   damage
        TypeId   contains
        uint32_t extent

#cdef extern from 'xtcdata/xtc/src/TransitionId.cc' namespace "XtcData":
#    pass

cdef extern from 'xtcdata/xtc/TransitionId.hh' namespace "XtcData":
    cdef cppclass TransitionId:
        enum Value: ClearReadout, Reset, Configure, Unconfigure, BeginRun, EndRun, BeginStep, EndStep, Enable, Disable, SlowUpdate, Unused_11, L1Accept = 12, NumberOf
        @staticmethod
        const char* name(Value id)

cdef extern from 'xtcdata/xtc/TimeStamp.hh' namespace "XtcData":
    cdef cppclass TimeStamp:
        TimeStamp() except +
        unsigned value() const
        unsigned seconds() const
        unsigned nanoseconds() const

cdef extern from 'xtcdata/xtc/Dgram.hh' namespace "XtcData":
    cdef cppclass Dgram:
        Dgram() except +
        uint16_t readoutGroups() const
        TransitionId.Value service() const
        TimeStamp time
        uint32_t env
        Xtc xtc

cdef extern from 'psdaq/service/EbDgram.hh' namespace "Pds":
    cdef cppclass PulseId:
        PulseId(uint64_t value) except +
        uint64_t pulseId() const

    cdef cppclass EbDgram:
        EbDgram(const PulseId& pulseId, const Dgram& dgram) except +
        PulseId pulseId
        Dgram dgram
