from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer
from libc.stdint cimport uint32_t, uint64_t
from dgramlite cimport Xtc, Sequence, Dgram
from psana.dgrampy import PyDgram

cdef class MyPyBuffer:
    cdef Py_buffer buf
    cdef uint64_t offset
    cdef uint64_t _view_size

    def __init__(self):
        self.offset = 0         # The beginning of the buffer and gets updated when we read a dgram
        self._view_size = 0

    def get_buffer(self, view):
        PyObject_GetBuffer(view, &(self.buf), PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
        self._view_size = memoryview(view).nbytes

    def dgrams(self):
        """A generator that keeps yielding a dgram from view"""
        cdef Dgram* dg
        cdef uint64_t payload, dgram_size, ts
        cdef char* view_ptr

        while self.offset < self._view_size:
            # Need to recast buf everytime we need to use it (not sure why)
            view_ptr = <char *>self.buf.buf
            view_ptr += self.offset

            dg = <Dgram *>(view_ptr)
            payload = dg.xtc.extent - sizeof(Xtc)
            ts = <uint64_t>dg.seq.high << 32 | dg.seq.low
            
            # Wrap the pointers to create a PyDgram obect.
            # Note that bufEnd just points to the end of dgram so this
            # dgram cannot be modified (no space to append anything to).
            pycap_dg = PyCapsule_New(<void *>dg, "dgram", NULL)
            dgram_size = sizeof(Dgram) + payload
            pydg = PyDgram(pycap_dg, dgram_size)

            # Update offset for next read
            self.offset += dgram_size
            yield pydg

    def free_buffer(self):
        PyBuffer_Release(&(self.buf))

    @property
    def view_size(self):
        return self._view_size


