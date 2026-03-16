from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE
from libc.stdint cimport uint32_t
cimport psdaq.trigger.CubeConfigDgram as dgram
import EbDgram as edg
import EbDgram as edg

cdef class CubeConfigDgram:
    cdef Py_buffer buf
    cdef dgram.CubeConfigDgram* cptr
    cdef int _bufOwner

    def __cinit__(self, view = None, nbins = 1, json_str = None):
        self._bufOwner = view is not None
        if self._bufOwner:
            #  Inheritance is not provided
            ebdg = edg.EbDgram(view)
            ebdg.xtc.alloc(8)
            #
            PyObject_GetBuffer(view, &self.buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
            view_ptr = <char *>self.buf.buf
            self.cptr = <dgram.CubeConfigDgram *>(view_ptr)
            self.cptr.bins         (nbins)
            if json_str:
               self.cptr.appendJson(json_str.encode("utf-8"))

    def __dealloc__(self):
        if self._bufOwner:
            PyBuffer_Release(&self.buf)

