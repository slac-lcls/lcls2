from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE
from libc.stdint cimport uint32_t
cimport psdaq.trigger.TimingTebData as ttt

cdef class TimingTebData():
    cdef Py_buffer buf
    cdef ttt.TimingTebData* cptr
    cdef int _bufOwner

    def __cinit__(self, view = None):
        self._bufOwner = view is not None
        if self._bufOwner:
            if PyObject_GetBuffer(view, &self.buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)<0:
               print('Error extracting TimingTebData buffer')
            self.cptr = <ttt.TimingTebData *>(self.buf.buf)

    def __dealloc__(self):
        if self._bufOwner:
            PyBuffer_Release(&self.buf)

    @property
    def eventcodes(self):
        rval = list()
        for i in range(9):
            e = self.cptr.eventcodes[i]
            for j in range(32):
                if e & (1<<j):
                   rval.append(i*32+j)
        return rval
