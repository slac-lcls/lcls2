from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE
from libc.stdint cimport int32_t, uint8_t
cimport psdaq.trigger.HrEncoderTebData as ttt

cdef class HrEncoderTebData():
    cdef Py_buffer buf
    cdef ttt.HrEncoderTebData* cptr
    cdef int _bufOwner

    def __cinit__(self, view = None):
        self._bufOwner = view is not None
        if self._bufOwner:
            if PyObject_GetBuffer(view, &self.buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)<0:
               print('Error extracting HrEncoderTebData buffer')
            self.cptr = <ttt.HrEncoderTebData *>(self.buf.buf)

    def __dealloc__(self):
        if self._bufOwner:
            PyBuffer_Release(&self.buf)

    def position(self):
        return self.cptr.m_position

    def encErrCnt(self):
        return self.cptr.m_encErrCnt

    def missedTrigCnt(self):
        return self.cptr.m_missedTrigCnt

    def latches(self):
        return self.cptr.m_latches

