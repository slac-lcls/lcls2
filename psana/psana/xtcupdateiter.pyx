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

class DataType:
    """ This list has to match Name.DataType c++ 
    but it doesn't make sense to wrap it ..."""
    UINT8   = 0
    UINT16  = 1
    UINT32  = 2
    UINT64  = 3
    INT8    = 4
    INT16   = 5
    INT32   = 6
    INT64   = 7
    FLOAT   = 8
    DOUBLE  = 9

cdef class PyNewDef:
    cdef NewDef* cptr

    def __init__(self):
        self.cptr = new NewDef()
        
    def show(self):
        self.cptr.show()

    def add(self, name, dtype, rank):
        self.cptr.add(name.encode(), dtype, rank)


cdef class PyXtc():
    cdef Xtc* cptr

    # No constructor - the Xtc ptr gets assigned elsewhere.

    def sizeofPayload(self):
        return self.cptr.sizeofPayload()
    
cdef class PyDgram():
    cdef Dgram* cptr

    # No constructor - the Dgram ptr gets assigned elsewhere.
    
    def get_pyxtc(self):
        pyxtc = PyXtc()
        pyxtc.cptr = &(self.cptr.xtc)
        return pyxtc

    def get_size(self):
        return sizeof(Dgram)

cdef class PyXtcUpdateIter():
    cdef XtcUpdateIter* cptr
    _numWords = 3 # no. of print-out elements for an array
    
    def __cinit__(self):
        self.cptr = new XtcUpdateIter(self._numWords)

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

    def add_names(self, PyXtc pyxtc, detdef, PyNewDef pynewdef):
        # Passing string to c needs utf-8 encoding
        detName = detdef.name.encode()
        detType = detdef.dettype.encode()
        detId   = detdef.detid.encode()
        algName = detdef.alg.name.encode()
        
        # Dereference in cython with * is not allowed. You can either use [0] index or
        # from cython.operator import dereference
        # cdef Xtc xtc = dereference(pyxtc.cptr)
        self.cptr.addNames(pyxtc.cptr[0], detName, detType, detId,
                detdef.nodeId, detdef.namesId, detdef.segment,
                algName, detdef.alg.major, detdef.alg.minor, detdef.alg.micro,
                pynewdef.cptr[0])

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
