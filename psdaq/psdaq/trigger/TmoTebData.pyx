from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE
from libc.stdint cimport uint32_t
cimport psdaq.trigger.TmoTebData as ttd

cdef class TmoTebData():
    cdef Py_buffer buf
    cdef ttd.TmoTebData* cptr
    cdef int _bufOwner

    def __cinit__(self, view = None):
        self._bufOwner = view is not None
        if self._bufOwner:
            PyObject_GetBuffer(view, &self.buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
            view_ptr = <char *>self.buf.buf
            self.cptr = <ttd.TmoTebData *>(view_ptr)

    def __dealloc__(self):
        if self._bufOwner:
            PyBuffer_Release(&self.buf)

    @property
    def write(self):
        return self.cptr.write

    @property
    def monitor(self):
        return self.cptr.monitor

