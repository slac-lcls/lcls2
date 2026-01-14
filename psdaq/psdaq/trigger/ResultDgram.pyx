from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE
from libc.stdint cimport uint32_t
cimport psdaq.trigger.ResultDgramw as dgram

cdef class ResultDgram():
    cdef Py_buffer buf
    cdef dgram.ResultDgram* cptr
    cdef int _bufOwner

    def __cinit__(self, view = None, persist = True, monitor = -1):
        self._bufOwner = view is not None
        if self._bufOwner:
            PyObject_GetBuffer(view, &self.buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
            view_ptr = <char *>self.buf.buf
            self.cptr = <dgram.ResultDgram *>(view_ptr)
            self.cptr.persist(persist)
            self.cptr.monitor(monitor)

    def __dealloc__(self):
        if self._bufOwner:
            PyBuffer_Release(&self.buf)

    def persist(self):
        return self.cptr.persist()

    def persist(self, value):
        return self.cptr.persist(value)

    def monitor(self):
        return self.cptr.monitor()

    def monitor(self, value):
        return self.cptr.monitor(value)

