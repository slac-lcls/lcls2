from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE
cimport psdaq.trigger.EbDgram as dgram
from libc.stdint cimport uint64_t

cdef class Src():
    cdef dgram.Src* cptr

    # No constructor - the Dgram ptr gets assigned elsewhere.

    def value(self):
        return self.cptr.value()

cdef class Damage():
    cdef dgram.Damage* cptr

    # No constructor - the Dgram ptr gets assigned elsewhere.

    def value(self):
        return self.cptr.value()

cdef class TypeId():
    cdef dgram.TypeId* cptr

    # No constructor - the Dgram ptr gets assigned elsewhere.

    def value(self):
        return self.cptr.value()

cdef class Xtc():
    cdef dgram.Xtc* cptr

    # No constructor - the Dgram ptr gets assigned elsewhere.

    def payload(self):
        # (char*) without an explicit size will be treated 
        # as a null-terminated string
        return self.cptr.payload()[:self.cptr.sizeofPayload()]

    def sizeofPayload(self):
        return self.cptr.sizeofPayload()

    @property
    def src(self):
        the_src = Src()
        the_src.cptr = &(self.cptr.src)
        return the_src

    @property
    def damage(self):
        the_damage = Damage()
        the_damage.cptr = &(self.cptr.damage)
        return the_damage

    @property
    def contains(self):
        the_typeId = TypeId()
        the_typeId.cptr = &(self.cptr.contains)
        return the_typeId

    @property
    def extent(self):
        return self.cptr.extent

cdef class TimeStamp():
    cdef dgram.TimeStamp* cptr

    # No constructor - the Dgram ptr gets assigned elsewhere.

    def value(self):
        return self.cptr.value()

    def seconds(self):
        return self.cptr.seconds()

    def nanoseconds(self):
        return self.cptr.nanoseconds()

cdef class EbDgram():
    cdef Py_buffer buf
    cdef dgram.EbDgram* cptr
    cdef int _bufOwner

    def __cinit__(self, view = None):
        self._bufOwner = view is not None
        if self._bufOwner:
            PyObject_GetBuffer(view, &self.buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
            self.cptr = <dgram.EbDgram *>(self.buf.buf)

    def __dealloc__(self):
        if self._bufOwner:
            PyBuffer_Release(&self.buf)

    def pulseId(self):
        the_pulseId = <dgram.PulseId*>(self.cptr)
        #print "*** EbDgram.pyx: &pulseId {0:x}".format(<unsigned long>the_pulseId)
        #print "*** EbDgram.pyx:  pulseId {0:x}".format(<unsigned long>the_pulseId.pulseId())
        return the_pulseId.pulseId()

    @property
    def time(self):
        the_time = TimeStamp()
        the_dgram = <dgram.Dgram*>(self.cptr)
        the_time.cptr = &(the_dgram.time)
        return the_time

    @property
    def env(self):
        the_dgram = <dgram.Dgram*>(self.cptr)
        return the_dgram.env

    @property
    def xtc(self):
        the_xtc =  Xtc()
        the_dgram = <dgram.Dgram*>(self.cptr)
        the_xtc.cptr = &(the_dgram.xtc)
        return the_xtc

    def readoutGroups(self):
        the_dgram = <dgram.Dgram*>(self.cptr)
        return the_dgram.readoutGroups()

    def sizeofPayload(self):
        the_dgram = <dgram.Dgram*>(self.cptr)
        return the_dgram.xtc.sizeofPayload()

    def service(self):
        the_dgram = <dgram.Dgram*>(self.cptr)
        return the_dgram.service()
