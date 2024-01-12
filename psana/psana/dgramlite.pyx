from dgramlite cimport Xtc, Sequence, Dgram
from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE
from libc.stdint cimport uint64_t

cdef class DgramLite:
    cdef uint64_t payload
    cdef uint64_t timestamp 
    cdef unsigned service
    
    def __init__(self, view):
        """Create Dgram (light-weight version) from a given buffer object."""
        cdef Dgram* d
        cdef char* view_ptr
        cdef Py_buffer buf
        
        PyObject_GetBuffer(view, &buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
        view_ptr = <char *>buf.buf
        d = <Dgram *>(view_ptr)
        self.payload = d.xtc.extent - sizeof(Xtc)
        self.timestamp = <uint64_t>d.seq.high << 32 | d.seq.low
        self.service = (d.env>>24)&0xf
        self.env = d.env
        PyBuffer_Release(&buf)

    @property
    def payload(self):
        return self.payload

    @property
    def timestamp(self):
        return self.timestamp

    @property
    def service(self):
        return self.service

    @property
    def keepraw(self):
        return (self.env>>6)&1


