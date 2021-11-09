from libc.string    cimport memcpy
from libcpp.string  cimport string
from xtcupdateiter  cimport XtcUpdateIter

import os

class AlgDef:
    def __init__(self, name, major, minor, micro):
        self.name   = name
        self.major  = major
        self.minor  = minor
        self.micro  = micro

class DetectorDef:
    def __init__(self, name, alg, dettype, detid,
            nodeId, namesId, segment):
        self.name       = name
        self.alg        = alg
        self.dettype    = dettype
        self.detid      = detid
        self.nodeId     = nodeId
        self.namesId    = namesId
        self.segment    = segment

cdef class PyXtc():
    cdef Xtc* cptr

    def sizeofPayload(self):
        return self.cptr.sizeofPayload()
    
cdef class PyDgram():
    cdef Dgram* cptr

    def get_pyxtc(self):
        pyxtc = PyXtc()
        pyxtc.cptr = &(self.cptr.xtc)
        return pyxtc

    def get_size(self):
        return sizeof(Dgram)

cdef class PyXtcUpdateIter():
    cdef unsigned _numWords
    cdef XtcUpdateIter* cptr
    
    def __cinit__(self, unsigned numWords):
        self._numWords = numWords
        self.cptr = new XtcUpdateIter(numWords)

    def process(self, PyXtc pyxtc):       
        self.cptr.process(pyxtc.cptr)

    def iterate(self, PyXtc pyxtc):
        self.cptr.iterate(pyxtc.cptr)

    def get_buf(self):
        cdef char[:] buf
        if self.cptr.get_bufsize() > 0:
            buf = <char [:self.cptr.get_bufsize()]>self.cptr.get_buf()
            return buf
        else:
            return memoryview(bytearray()) 

    def copy_dgram(self, PyDgram pydg):
        print(f'sizeof(Dgram): {sizeof(Dgram)}')
        self.cptr.copy2buf(<char *>pydg.cptr, sizeof(Dgram))

    def add_names(self, PyXtc pyxtc, detdef):
        detName = detdef.name.encode()
        
        # Dereference in cython with * is not allowed. You can either use [0] index or
        # from cython.operator import dereference
        # cdef Xtc xtc = dereference(pyxtc.cptr)
        self.cptr.addNames(pyxtc.cptr[0], detName, detdef.nodeId, detdef.namesId, detdef.segment)

    def add_data(self, PyXtc pyxtc, unsigned nodeId, unsigned namesId):
        self.cptr.addData(pyxtc.cptr[0], nodeId, namesId)


cdef class PyXtcFileIterator():
    cdef XtcFileIterator* cptr

    def __cinit__(self, int fd, size_t maxDgramSize):
        self.cptr = new XtcFileIterator(fd, maxDgramSize)

    def next(self):
        cdef Dgram* dg
        dg = self.cptr.next()
        pydg = PyDgram()
        pydg.cptr = dg
        return pydg
