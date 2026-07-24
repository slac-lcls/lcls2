from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE
from libc.stdint cimport uint32_t
cimport psdaq.trigger.WindowResultDgramw as dgram
import psdaq.EbDgram as edg

cdef class WindowResultDgram:
    cdef Py_buffer buf
    cdef dgram.WindowResultDgram* cptr
    cdef int _bufOwner

    def __cinit__(self, view = None, persist = True, monitor = -1, win_add = [], win_flush = []):
        self._bufOwner = view is not None
        if self._bufOwner:
            #  Inheritance is not provided
            ebdg = edg.EbDgram(view)
            ebdg.xtc.alloc(8)
            #
            PyObject_GetBuffer(view, &self.buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
            view_ptr = <char *>self.buf.buf
            self.cptr = <dgram.WindowResultDgram *>(view_ptr)
            self.cptr.persist      (persist)
            self.cptr.monitor      (monitor)
            win_add_m = 0
            for w in win_add:
                win_add_m |= 1<<w
            self.cptr.updateAdd    (win_add_m)
            win_flush_m = 0
            for w in win_flush:
                win_flush_m |= 1<<w
            self.cptr.updateRecord (win_flush_m)
            self.cptr.updateMonitor(win_flush_m)
            self.cptr.flush        (win_flush_m)

    def __dealloc__(self):
        if self._bufOwner:
            PyBuffer_Release(&self.buf)

    def data(self):
        return self.cptr.data()

    def persist(self):
        return self.cptr.persist()

    def monitor(self):
        return self.cptr.monitor()

"""
    def updatedAdd(self):
        return self.cptr.updateAdd()

    def updateRecord(self):
        return self.cptr.updateRecord()

    def updateMonitor(self):
        return self.cptr.updateMonitor()

    def flush(self):
        return self.cptr.flush()
"""