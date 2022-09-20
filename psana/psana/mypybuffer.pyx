from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer
from libc.stdint cimport uint32_t, uint64_t
from dgramlite cimport Xtc, Sequence, Dgram
from psana.dgramedit import PyDgram

cdef class MyPyBuffer:
    cdef Py_buffer buf
    cdef uint64_t offset
    cdef uint64_t view_size
    cdef Dgram* dg               # This is set when next(dgrams()) is called

    def __init__(self, view):
        PyObject_GetBuffer(view, &(self.buf), PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
        self.view_size = memoryview(view).nbytes
        self.offset = 0         # The beginning of the buffer and gets updated when we read a dgram
    
    def __dealloc__(self):
        PyBuffer_Release(&(self.buf))

    def as_pydg(self):
        cdef uint64_t payload, dgram_size
        cdef Dgram* dg
        dg = self.dg
        payload = dg.xtc.extent - sizeof(Xtc)
        
        # Wrap the pointers to create a PyDgram obect.
        # Note that bufEnd just points to the end of dgram so this
        # dgram cannot be modified (no space to append anything to).
        pycap_dg = PyCapsule_New(<void *>dg, "dgram", NULL)
        dgram_size = sizeof(Dgram) + payload
        pydg = PyDgram(pycap_dg, dgram_size)
        return pydg
    
    @property
    def dgram(self):
        return self.as_pydg()

    def dgrams(self):
        """A generator that keeps yielding a PyDgram from view.
        We set self.dg (Dgram *) so that we can also retrieve the most
        recent yieled PyDgram (call as_pydg to wrapt the ptr). 
        """
        cdef uint64_t payload, dgram_size
        cdef char* view_ptr

        while self.offset < self.view_size:
            # Need to recast buf everytime we need to use it (not sure why)
            view_ptr = <char *>self.buf.buf
            view_ptr += self.offset

            self.dg = <Dgram *>(view_ptr)
            payload = self.dg.xtc.extent - sizeof(Xtc)
            dgram_size = sizeof(Dgram) + payload

            # Update offset for next read
            self.offset += dgram_size
            
            yield self.as_pydg()

    def rewind(self):
        """Moves the offset back one dgram."""
        self.offset -= self.dgram.size()

    @property
    def size(self):
        return self.view_size

    @property
    def offset(self):
        return self.offset
