from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE
from libc.stdint cimport uint32_t, uint8_t
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

    def has_destination(self,destn):
        return (self.cptr.ebeamDestn&0x80) and (self.cptr.ebeamDestn&0x7f)==destn

    def has_eventcode(self,code):
        return (self.cptr.eventcodes[code>>5]>>(code&0x1f))&1

    def eventcodes(self):
        rval = list()
        for i in range(9):
            e = self.cptr.eventcodes[i]
            for j in range(32):
                if e & (1<<j):
                   rval.append(i*32+j)
        return rval

    def eventcodes_to_int(self, low, high):
        ilo = low>>5
        ihi = high>>5
        rval = self.cptr.eventcodes[ilo] >> (low - 32*ilo)
        for i in range(ilo+1,ihi+1):
            e = self.cptr.eventcodes[i]
            rval |= e << (32*i - low)
        rval &= (1<<(high-low+1))-1
        return rval

