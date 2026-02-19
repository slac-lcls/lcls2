from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE
cimport psdaq.trigger.Piranha4TTTebData as ttt

cdef class Piranha4TTTebData():
    cdef Py_buffer buf
    cdef ttt.Piranha4TTTebData* cptr
    cdef int _bufOwner

    def __cinit__(self, view = None):
        self._bufOwner = view is not None
        if self._bufOwner:
            if PyObject_GetBuffer(view, &self.buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)<0:
               print('Error extracting Piranha4TTTebData buffer')
            self.cptr = <ttt.Piranha4TTTebData *>(self.buf.buf)

    def __dealloc__(self):
        if self._bufOwner:
            PyBuffer_Release(&self.buf)

    def valid(self):
        return self.cptr.m_ampl >= 0

    def ampl(self):
        return self.cptr.m_ampl

    def fltpos(self):
        return self.cptr.m_fltpos

    def fltpos_ps(self):
        return self.cptr.m_fltpos_ps

    def fltpos_fwhm(self):
        return self.cptr.m_fltpos_fwhm

    def amplnxt(self):
        return self.cptr.m_amplnxt

    def refampl(self):
        return self.cptr.m_refampl

