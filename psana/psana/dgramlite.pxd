cdef struct Xtc:
    int junks[4]
    unsigned extent

cdef struct Sequence:
    int junks[2]
    unsigned low
    unsigned high

cdef struct Dgram:
    Sequence seq
    int junks[4]
    Xtc xtc

