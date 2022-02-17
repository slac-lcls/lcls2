from libc.string    cimport memcpy
from libc.stdlib    cimport malloc, free
from libcpp.string  cimport string
from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE
from psana.dgrampy  cimport *
from cpython cimport array
import array
from psana.psexp import TransitionId
import numpy as np
import numbers


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
    {UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64, FLOAT, DOUBLE, 
    CHARSTR, ENUMVAL, ENUMDICT}
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
    CHARSTR = 10
    ENUMVAL = 11
    ENUMDICT= 12
    nptypes = {
                0: np.uint8,
                1: np.uint16,
                2: np.uint32,
                3: np.uint64,
                4: np.int8, 
                5: np.int16,
                6: np.int32,
                7: np.int64,
                8: np.float32,
                9: np.float64,
    }

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

    def dec_extent(self, uint32_t removed_size):
        self.cptr.xtc.extent -= removed_size

    def get_payload_size(self):
        return self.cptr.xtc.sizeofPayload()


cdef class PyXtcUpdateIter():
    cdef XtcUpdateIter* cptr
    _numWords = 6 # no. of print-out elements for an array
    
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

    def clear_buf(self):
        self.cptr.clear_buf()

    def copy(self, PyDgram pydg):
        self.cptr.copy(pydg.cptr)

    def updatetimestamp(self, PyDgram pydg, sec, nsec):
        self.cptr.updateTimeStamp(pydg.cptr[0], sec, nsec)

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

    def setstring(self, PyDataDef pydatadef, datadef_name, str_data):
        self.cptr.setString(str_data.encode(), pydatadef.cptr[0], datadef_name.encode())
    
    def setvalue(self, namesdef, PyDataDef pydatadef, datadef_name, data):
        """Sets scalar value to the given `datadef_name`.
        
        Converts (any) given scalar `data` to the specified known dtype
        using numpy array format. The buffer is passed in (and later converted
        to the correct dtype) as char*. 
        """
        cdef int dtypeNo = pydatadef.cptr.getDtype(datadef_name.encode())
        
        d_arr = np.array([data], dtype=DataType.nptypes[dtypeNo])
        cdef char* data_ptr
        cdef Py_buffer data_pybuf
        PyObject_GetBuffer(d_arr, &data_pybuf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
        data_ptr = <char *>data_pybuf.buf
        self.cptr.setValue(namesdef.nodeId, namesdef.namesId, 
                           data_ptr, pydatadef.cptr[0], datadef_name.encode())
        PyBuffer_Release(&data_pybuf)

    def adddata(self, namesdef, PyDataDef pydatadef, 
            datadef_name, unsigned[:] shape, data):
        cdef unsigned* shape_ptr
        cdef Py_buffer shape_pybuf
        PyObject_GetBuffer(shape, &shape_pybuf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
        shape_ptr = <unsigned *>shape_pybuf.buf

        cdef char* data_ptr
        cdef Py_buffer data_pybuf
        PyObject_GetBuffer(data, &data_pybuf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
        data_ptr = <char *>data_pybuf.buf

        self.cptr.addData(namesdef.nodeId, namesdef.namesId, shape_ptr, 
                data_ptr, pydatadef.cptr[0], datadef_name.encode())
        
        PyBuffer_Release(&shape_pybuf)
        PyBuffer_Release(&data_pybuf)

    def removedata(self, det_name, alg_name):
        self.cptr.setFilter(det_name.encode(), alg_name.encode())
    
    def createTransition(self, transId, timestamp_val):
        counting_timestamps = 1
        if timestamp_val == -1:
            counting_timestamps = 0
            timestamp_val = 0 

        pydgram = PyDgram()
        pydgram.cptr = &(self.cptr.createTransition(transId, 
                counting_timestamps, timestamp_val))
        return pydgram
    
    def get_removed_size(self):
        return self.cptr.get_removed_size()



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

    print(f"\nCYTHON adddata")
    print(f"CYTHON call createdata")
    uiter.createdata(pyxtc, namesdef)

    print(f"CYTHON call adddata/setvalue/setstring")
    for datadef_name, data in datadict.items():
        # Handle scalar types (string or number)
        if np.ndim(data) == 0:
            if isinstance(data, numbers.Number):
                uiter.setvalue(namesdef, pydatadef, datadef_name, data)
            else:
                uiter.setstring(pydatadef, datadef_name, data)
            continue
        
        # Handle array type
        array.zero(shape)
        if data.dtype.type is np.str_:
            # For strings, from what I understand is each string is
            # an array so for MaxRank=5 we can only store up to 5 strings.
            # shape[i] is no. of bytes for string i.
            
            assert len(data.shape) == 1, "Only support writing 1D string array"
            
            str_as_byte = bytearray()
            for i in range(data.shape[0]):
                data_encode = data[i].encode()
                str_as_byte.extend(data_encode)
                shape[i] = len(data_encode)
            
            uiter.adddata(namesdef, pydatadef, datadef_name, shape, str_as_byte)
        else:
            for i in range(len(shape)):
                if i < len(data.shape):
                    shape[i] = data.shape[i]
            uiter.adddata(namesdef, pydatadef, datadef_name, shape, data)
    print(f'CYTHON DONE addata')

def removedata(det_name, alg_name):
    uiter.removedata(det_name, alg_name)

def datadef(datadict):
    datadef = PyDataDef()
    for key, val in datadict.items():
        datadef.add(key, val[0], val[1])
    return datadef

def get_buf():
    return uiter.get_buf()

def save(PyDgram pydg):
    """ Copies Names and ShapesData to _tmpbuf and update
    parent dgram extent to new size (if some ShapesData
    were removed. The parent dgram and _tmpbuf are then
    copied to _buf as one new event. """
    # Copies Names and ShapesData (also applies remove for 
    # ShapesData) to _tmpbuf
    pyxtc = pydg.get_pyxtc()
    uiter.iterate(pyxtc)

    # Update parent dgram with new size
    cdef uint32_t removed_size = uiter.get_removed_size()
    if removed_size > 0:
        print(f'pydg payload before:{pydg.get_payload_size()}')
        pydg.dec_extent(removed_size)
        print(f'pydg payload after:{pydg.get_payload_size()}')

    # Copy updated parent dgram and _tmpbuf to _buf 
    uiter.copy(pydg)

def clearbuf():
    uiter.clear_buf()

def updatetimestamp(PyDgram pydg, sec, nsec):
    uiter.updatetimestamp(pydg, sec, nsec)
