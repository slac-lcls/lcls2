from libc.stdint cimport uint64_t, uint32_t, uint16_t, uint8_t
from libc.string cimport memcpy

cdef struct Src:
    uint32_t src

cdef struct Damage:
    uint16_t damage

cdef struct TypeId:
    uint16_t value

cdef struct Xtc:
    Src src
    Damage damage
    TypeId contains
    uint32_t extent

cdef struct Sequence:
    uint32_t low
    uint32_t high

cdef struct Dgram:
    Sequence seq
    uint32_t env
    Xtc xtc
