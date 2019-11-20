from libc.stdint cimport uint32_t, uint64_t

cdef struct Xtc:
    int junks[2]
    uint32_t extent

cdef struct Sequence:
    uint64_t pulse_id
    uint32_t low
    uint32_t high

cdef struct Dgram:
    Sequence seq
    uint32_t env
    Xtc xtc

