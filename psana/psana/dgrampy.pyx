from libc.string    cimport memcpy
from libcpp.string  cimport string
from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE
from psana.dgrampy  cimport *
from cpython cimport array
import array
from psana.psexp import TransitionId


class AlgDef:
    def __init__(self, name, major, minor, micro):
        self.name   = name
        self.major  = major
        self.minor  = minor
        self.micro  = micro

class DetectorDef:
    def __init__(self, name, dettype, detid):
        self.name       = name
        self.dettype    = dettype
        self.detid      = detid

class NamesDef:
    def __init__(self, nodeId, namesId):
        self.nodeId = nodeId
        self.namesId = namesId


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

cdef class PyDataDef:
    cdef DataDef* cptr

    def __init__(self):
        self.cptr = new DataDef()
        
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

    def names(self, PyXtc pyxtc, detdef, algdef, PyDataDef pydatadef, 
            nodeId=None, namesId=None, segment=None):
        # Passing string to c needs utf-8 encoding
        detName = detdef.name.encode()
        detType = detdef.dettype.encode()
        detId   = detdef.detid.encode()
        algName = algdef.name.encode()

        if not nodeId: nodeId = 1
        if not namesId: namesId = 1
        if not segment: segment = 0
        
        # Dereference in cython with * is not allowed. You can either use [0] index or
        # from cython.operator import dereference
        # cdef Xtc xtc = dereference(pyxtc.cptr)
        self.cptr.addNames(pyxtc.cptr[0], detName, detType, detId,
                nodeId, namesId, segment,
                algName, algdef.major, algdef.minor, algdef.micro,
                pydatadef.cptr[0])
        return NamesDef(nodeId, namesId)

    def createdata(self, PyXtc pyxtc, namesdef):
        self.cptr.createData(pyxtc.cptr[0], namesdef.nodeId, namesdef.namesId)

    def adddata(self, PyXtc pyxtc, namesdef, PyDataDef pydatadef, 
            datadef_name, unsigned[:] shape, char[:,:] data):
        cdef unsigned* shape_ptr
        cdef Py_buffer shape_pybuf
        PyObject_GetBuffer(shape, &shape_pybuf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
        shape_ptr = <unsigned *>shape_pybuf.buf

        cdef char* data_ptr
        cdef Py_buffer data_pybuf
        PyObject_GetBuffer(data, &data_pybuf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
        data_ptr = <char *>data_pybuf.buf

        self.cptr.addData(pyxtc.cptr[0], namesdef.nodeId, namesdef.namesId, shape_ptr, 
                data_ptr, pydatadef.cptr[0], datadef_name.encode())
        
        PyBuffer_Release(&shape_pybuf)
        PyBuffer_Release(&data_pybuf)
    
    def createTransition(self, transId, timestamp_val):
        counting_timestamps = 1
        if timestamp_val == -1:
            counting_timestamps = 0
            timestamp_val = 0 

        pydgram = PyDgram()
        pydgram.cptr = &(self.cptr.createTransition(transId, 
                counting_timestamps, timestamp_val))
        return pydgram



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

"""
x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-
List of convenient functions that allow users to access
above classes w/o typing them out.
x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-
"""

# Main class used for creating and updating xtc data
uiter = PyXtcUpdateIter()

def config(ts=-1):
    cfg = uiter.createTransition(TransitionId.Configure, ts)
    return cfg

def dgram(ts=-1):
    dg = uiter.createTransition(TransitionId.L1Accept, ts)
    return dg

def alg(algname, major, minor, macro):
    return AlgDef(algname, major, minor, macro)

def det(detname, dettype, detid):
    return DetectorDef(detname, dettype, detid)

def names(PyDgram pydg, detdef, algdef, PyDataDef pydatadef, 
        nodeId=None, namesId=None, segment=None):
    pyxtc = pydg.get_pyxtc()
    return uiter.names(pyxtc, detdef, algdef, pydatadef, 
            nodeId, namesId, segment)

def adddata(PyDgram pydg, namesdef, PyDataDef pydatadef, datadict):
    pyxtc = pydg.get_pyxtc()
    cdef array.array shape = array.array('I', [0,0,0,0,0])
    cdef int i

    uiter.createdata(pyxtc, namesdef)

    for datadef_name, data in datadict.items():
        for i in range(len(shape)):
            if i < len(data.shape):
                shape[i] = data.shape[i]
            else:
                shape[i] = 0
        uiter.adddata(pyxtc, namesdef, pydatadef, datadef_name, shape, data)

def datadef(datadict):
    datadef = PyDataDef()
    for key, val in datadict.items():
        datadef.add(key, val[0], val[1])
    return datadef


def copy_dgram(PyDgram pydg):
    uiter.copy_dgram(pydg)

def get_buf():
    return uiter.get_buf()

def iterate(PyDgram pydg):
    pyxtc = pydg.get_pyxtc()
    uiter.iterate(pyxtc)
