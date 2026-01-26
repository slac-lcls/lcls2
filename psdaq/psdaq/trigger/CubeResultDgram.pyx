from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE
from libc.stdint cimport uint32_t
cimport psdaq.trigger.CubeResultDgram as dgram

cdef class CubeResultDgram:
    cdef Py_buffer buf
    cdef dgram.CubeResultDgram* cptr
    cdef int _bufOwner

    def __cinit__(self, view = None, persist = True, monitor = -1, bin = 0, worker = 0, recordBin = False):
        self._bufOwner = view is not None
        if self._bufOwner:
            PyObject_GetBuffer(view, &self.buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
            view_ptr = <char *>self.buf.buf
            self.cptr = <dgram.CubeResultDgram *>(view_ptr)
            self.cptr.persist(persist)
            self.cptr.monitor(monitor)
            self.cptr.bin    (bin)
            self.cptr.worker (worker)
            self.cptr.record (recordBin)

    def __dealloc__(self):
        if self._bufOwner:
            PyBuffer_Release(&self.buf)

    def data(self):
        return self.cptr.data()

    def persist(self):
        return self.cptr.persist()

    def monitor(self):
        return self.cptr.monitor()

    def bin(self):
        return self.cptr.bin()

    def worker(self):
        return self.cptr.worker()

    def record(self):
        return self.cptr.record()
