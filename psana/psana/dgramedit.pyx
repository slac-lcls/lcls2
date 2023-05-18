from libc.string    cimport memcpy
from libc.stdlib    cimport malloc, free
from libcpp.string  cimport string
from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer
from psana.dgramedit  cimport *
from cpython cimport array
import array
import numpy as np
import numbers
import io

MAXBUFSIZE=64000000

# Need a different name to prevent clashing with cpp class
from psana.psexp import TransitionId as PyTransitionId


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
                10: str,
    }
    psana_types = {np.uint8:    0,
                   np.uint16:   1,
                   np.uint32:   2,
                   np.uint64:   3,
                   np.int8:     4,
                   np.int16:    5,
                   np.int32:    6,
                   np.int64:    7,
                   np.float32:  8,
                   np.float64:  9,
                   str:         10,
                  }
    def to_psana2(self, numpy_dtype):
        return self.psana_types[numpy_dtype]

    def to_numpy(self, psana_dtype):
        return self.nptypes[psana_dtype]

cdef class PyDataDef:
    cdef DataDef* cptr

    def __cinit__(self):
        self.cptr = new DataDef()

    def __dealloc__(self):
        del self.cptr

    def show(self):
        self.cptr.show()

    def add(self, name, dtype, rank):
        self.cptr.add(name.encode(), dtype, rank)

    def get_dtype(self, name):
        return self.cptr.getDtype(name.encode())

    def get_rank(self, name):
        return self.cptr.getRank(name.encode())

    def is_number(self, name):
        flag = False
        if self.cptr.getDtype(name.encode()) != DataType.psana_types[str]:
            flag = True
        return flag


cdef class PyCapsuleDgram():
    cdef Py_buffer oPybuf
    def __init__(self, buf):
        """ Wraps pointer of a buffer object"""
        PyObject_GetBuffer(buf, &(self.oPybuf), PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)

    def get(self):
        return PyCapsule_New(<char *>self.oPybuf.buf, "dgram", NULL)


cdef class PyXtc():
    cdef Xtc* cptr
    cdef const void* bufEnd             # See bufEnd in PyDgram

    def __init__(self, pycap_xtc, pycap_bufend):
        self.cptr = <Xtc*>PyCapsule_GetPointer(pycap_xtc,"xtc")
        self.bufEnd = PyCapsule_GetPointer(pycap_bufend, "bufEnd")

    def sizeofPayload(self):
        return self.cptr.sizeofPayload()

cdef class PyDgram():
    cdef Dgram* cptr

    # Limit identifier for the dgram to identify the end of the buffer. 
    # For reading dgram from a file, this is size() of XtcFileIterator.
    # For createTransition, this is BUFSIZE in XtcUpdateIter.
    cdef const void* bufEnd
    cdef uint64_t bufSize
    
    def __init__(self, pycap_dgram, bufsize):
        self.cptr = <Dgram*>PyCapsule_GetPointer(pycap_dgram,"dgram")
        self.bufSize = bufsize
        self.bufEnd = <char*>self.cptr + self.bufSize

    @property
    def pyxtc(self):
        # We can't pass c ptr to Python function (takes Python object).
        # Follow Valerio's work, here we store c ptr in PyCapsule.
        pycap_xtc = PyCapsule_New(<void *>&(self.cptr.xtc), "xtc", NULL)
        pycap_bufend = PyCapsule_New(<void *>(self.bufEnd), "bufEnd", NULL)
        pyxtc = PyXtc(pycap_xtc, pycap_bufend)
        return pyxtc

    def dec_extent(self, uint32_t removed_size):
        self.cptr.xtc.extent -= removed_size
        # The minimum value of extent is the size of Xtc. 
        err = f"Invalid extent value: {self.cptr.xtc.extent} (must be >= sizeof(Xtc): {sizeof(Xtc)})"
        assert self.cptr.xtc.extent >= sizeof(Xtc), err

    def get_payload_size(self):
        return self.cptr.xtc.sizeofPayload()

    def size(self):
        return sizeof(Dgram) + self.get_payload_size()

    def service(self):
        return self.cptr.service()

    def timestamp(self):
        return self.cptr.time.value()

    def as_memoryview(self):
        cdef uint64_t dgram_size = self.size()
        return <char [:dgram_size]><char*>self.cptr



cdef class PyXtcUpdateIter():
    cdef XtcUpdateIter* cptr
    _numWords = 6               # no. of print-out elements for an array
    cdef const void* bufEnd
    cdef Py_buffer oPybuf

    def __cinit__(self):
        self.cptr = new XtcUpdateIter(self._numWords)

    def __dealloc__(self):
        # Ric added this (see cinit above for an allocation of c++ object
        # as pointer)
        del self.cptr

    def process(self, PyXtc pyxtc):
        # Note on bufEnd, this pyxtc.bufEnd comes from the PyDgram
        # object that owns it. This is differnt from self.bufEnd which is used
        # in createTransition(). The value of pyxtc.bufEnd comes from
        # the externally defined value. In test_dgrampy.py example, this value
        # is the maxDgramSize of PyXtcFileIterator (used to iterate over
        # dgrams in the given xtc file). 
        self.cptr.process(pyxtc.cptr, pyxtc.bufEnd)

    def iterate(self, PyXtc pyxtc):
        self.cptr.iterate(pyxtc.cptr, pyxtc.bufEnd)

    def get_size(self):
        return self.cptr.getSize()

    def copy_parent(self, PyDgram pydg):
        self.cptr.copyParent(pydg.cptr)

    def set_outbuf(self, outbuf):
        cdef char* o_ptr
        PyObject_GetBuffer(outbuf, &(self.oPybuf), PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
        o_ptr = <char *>self.oPybuf.buf
        self.cptr.setOutput(o_ptr)
    
    def free_outbuf(self):
        # TODO: Find way toProtect when PyObject_GetBuffer is not called.
        PyBuffer_Release(&(self.oPybuf))

    def updatetimestamp(self, PyDgram pydg, timestamp_val):
        self.cptr.updateTimeStamp(pydg.cptr[0], timestamp_val)

    def names(self, PyDgram pydg, detdef, algdef, PyDataDef pydatadef,
            nodeId, namesId, segment):
        cdef PyXtc pyxtc = pydg.pyxtc

        # Passing string to c needs utf-8 encoding
        detName = detdef.name.encode()
        detType = detdef.dettype.encode()
        detId   = detdef.detid.encode()
        algName = algdef.name.encode()

        # For automatic assignment,
        # for namesId, increment the current value by 1.
        # for nodeId, use the current nodeId of this Configure.
        # for segment, use 0.
        if nodeId is None: nodeId = self.cptr.getNodeId()
        if namesId is None: namesId = self.cptr.getNextNamesId()
        if segment is None: segment = 0

        # Dereference in cython with * is not allowed. You can either use [0] index or
        # from cython.operator import dereference
        # cdef Xtc xtc = dereference(pyxtc.cptr)
        self.cptr.addNames(pyxtc.cptr[0], pydg.bufEnd, detName, detType, detId,
                nodeId, namesId, segment,
                algName, algdef.major, algdef.minor, algdef.micro,
                pydatadef.cptr[0])
        return NamesDef(nodeId, namesId)

    def createdata(self, PyDgram pydg, namesdef):
        cdef PyXtc pyxtc = pydg.pyxtc
        self.cptr.createData(pyxtc.cptr[0], pydg.bufEnd, namesdef.nodeId, namesdef.namesId)

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

    def set_filter(self, det_name, alg_name):
        self.cptr.setFilter(det_name.encode(), alg_name.encode())

    def createTransition(self, transId, timestamp_val, pycap_buf, bufsize):
        counting_timestamps = 1
        if timestamp_val == -1:
            counting_timestamps = 0
            timestamp_val = 0

        # Extract buffer ptr and set bufEnd
        cdef char* buf = <char *>PyCapsule_GetPointer(pycap_buf, "buf")
        cdef uint64_t bufSize = bufsize
        self.bufEnd = buf + bufSize
        
        # This returns Dgram and updates bufEnd to point to
        # buffer size of the dgram (defined in XtcUpdateIter).
        cdef Dgram* dg =  &(self.cptr.createTransition(transId,
                counting_timestamps, timestamp_val, buf))
        pycap_dg = PyCapsule_New(<void *>dg, "dgram", NULL)
        pydg = PyDgram(pycap_dg, <char*>self.bufEnd - <char*>dg)
        return pydg

    def get_removed_size(self):
        return self.cptr.getRemovedSize()

    def set_cfgwrite(self, int flag):
        self.cptr.setCfgWriteFlag(flag)

    def set_cfg(self, int flag):
        self.cptr.setCfgFlag(flag)

    def is_config(self):
        return self.cptr.isConfig()


cdef class PyXtcFileIterator():
    cdef XtcFileIterator* cptr

    def __cinit__(self, int fd, size_t maxDgramSize):
        self.cptr = new XtcFileIterator(fd, maxDgramSize)

    def __dealloc__(self):
        del self.cptr

    def next(self):
        cdef Dgram* dg
        dg = self.cptr.next()
        
        # Wrap the pointers and set the bufEnd limit to maxDgarmsize
        pycap_dg = PyCapsule_New(<void *>dg, "dgram", NULL)
        pydg = PyDgram(pycap_dg, self.cptr.size())
        return pydg

cdef class DgramEdit:
    cdef char* buf
    cdef PyXtcUpdateIter uiter
    cdef PyDgram pydg
    """ Main class that takes an existing dgram or creates
    new dgram when `TransitionId` is given and allow modifications
    on the dgram.
    """
    def __init__(self,
                 PyDgram pydg=None,
                 config=None,
                 transition_id=None,
                 ts=-1,
                 bufsize=MAXBUFSIZE):
        # We need to have only one PyXtcUpdateIter when working with
        # the same xtc. This class always read in config first and
        # populates NamesLookup, which is used by other dgrams.
        # Note that PyXtCUpdateIter takes bufsize. If this value is 0,
        # then the default bufsize (defined in XtcUpdateIter) is used.
        self.buf = NULL
        if pydg:
            if config:
                self.uiter = config.uiter
                self.uiter.set_cfg(False)
            else:
                # Assumes this is a config so we create new XtcUpdateIter here.
                self.uiter = PyXtcUpdateIter()
                self.uiter.set_cfg(True)
                # Iterates Configure without writing to buffer to get
                # current nodeId and namesId
                self.uiter.set_cfgwrite(False)
                self.uiter.iterate(pydg.pyxtc)

            self.pydg = pydg
        elif transition_id:
            if transition_id == PyTransitionId.Configure:
                self.uiter = PyXtcUpdateIter()
                self.uiter.set_cfg(True)
            else:
                assert config, "Missing config dgram"
                self.uiter = config.uiter
                self.uiter.set_cfg(False)
            self.buf = <char *>malloc(bufsize)
            pycap_buf = PyCapsule_New(<char *>self.buf, "buf", NULL)
            self.pydg = self.uiter.createTransition(transition_id, ts, pycap_buf, bufsize)
        else:
            raise IOError, "Unsupported input arguments"
    
    def __dealloc__(self):
        if self.buf != NULL:
            free(self.buf)

    def Detector(self, detdef, algdef, datadef_dict,
            nodeId=None, namesId=None, segment=None):
        """Returns Names object, which represents to a detector."""
        assert self.uiter.is_config() == 1, "Expect a Configure dgram."

        # Creates a datadef from data definition dictionary
        pydatadef = PyDataDef()
        dt = DataType()
        for key, (numpy_dtype, rank)  in datadef_dict.items():
            pydatadef.add(key, dt.to_psana2(numpy_dtype), rank)

        det = self.uiter.names(self.pydg, detdef, algdef, pydatadef,
                nodeId, namesId, segment)
        setattr(det, 'datadef', pydatadef)

        class Container:
            """Make datadef available as attributes of det.alg."""
            def __init__(self):
                # Save parent detector object here for later use
                setattr(self, '_det', det)
                # Wraps all fields in `pydatadef`
                for dtdef_name, _ in datadef_dict.items():
                    setattr(self, dtdef_name, None)

        setattr(det, algdef.name, Container())
        return det


    def adddata(self, data_container):
        cdef array.array shape = array.array('I', [0,0,0,0,0])
        cdef int i

        det = data_container._det
        self.uiter.createdata(self.pydg, det)

        pydatadef = det.datadef
        datadict = data_container.__dict__
        dt = DataType()

        for datadef_name, data in datadict.items():
            # Skips all reserved names
            if datadef_name.startswith('_'):
                continue

            assert data is not None, f"Missing data for '{datadef_name}'."

            # Check rank (dimension) of data (note that string is rank 1 in psana2).
            data_rank = np.ndim(data)
            if isinstance(data, str):
                data_rank = 1

            # Prepares error messages for incorrect ranks and types
            err_r = f"Incorrect rank for '{datadef_name}'. Expected: {pydatadef.get_rank(datadef_name)}."
            err_t = f"Incorrect type for '{datadef_name}'. Expected: {DataType.nptypes[pydatadef.get_dtype(datadef_name)]}."
            assert data_rank == pydatadef.get_rank(datadef_name), err_r


            # Handle scalar types (string or number)
            if np.ndim(data) == 0:
                if pydatadef.is_number(datadef_name):
                    assert isinstance(data, numbers.Number), err_t
                    self.uiter.setvalue(det, pydatadef, datadef_name, data)
                else:
                    assert isinstance(data, str), err_t
                    self.uiter.setstring(pydatadef, datadef_name, data)
            else:
                # Handle array type
                assert data.dtype.type != np.str_, f"Incorrect data for '{datadef_name}'. Array of string is not permitted. Use string instead (e.g. 'hello world')."
                assert pydatadef.get_dtype(datadef_name) == dt.to_psana2(data.dtype.type), err_t

                array.zero(shape)
                for i in range(len(shape)):
                    if i < len(data.shape):
                        shape[i] = data.shape[i]
                self.uiter.adddata(det, pydatadef, datadef_name, shape, data)
            # Clears data container for next event
            datadict[datadef_name] = None


    def removedata(self, det_name, alg_name):
        self.uiter.set_filter(det_name, alg_name)

    def save(self, out):
        """
        For L1Accept,
        Copies ShapesData to _outbuf and update
        parent dgram extent to new size (if some ShapesData
        were removed.

        For Configure,
        Calls iterate after setting cfgWriteFlag to True. This
        does two things:
        1. Iterating Names (writing is optional because we also
        just want to count NodeId and NamesId here). will also
        save to _outbuf
        2. Iterating ShapesData (writing is always) will save to
        _outbuf (for non-configure dgrams).
        """
        cdef PyXtc pyxtc = self.pydg.pyxtc
        is_config = False
        self.uiter.set_cfgwrite(False)
        if self.pydg.service() == PyTransitionId.Configure:
            is_config = True
            self.uiter.set_cfgwrite(True)
        
        # Set main output buffer to point to external buffer given by users
        self.uiter.set_outbuf(out)

        # Copies Names for Configure or ShapesData for L1Accept
        # (also applies remove for ShapesData).
        self.uiter.iterate(pyxtc)

        # Update parent dgram with new size
        cdef uint32_t removed_size = self.uiter.get_removed_size()
        if removed_size > 0:
            self.pydg.dec_extent(removed_size)

        # Copy updated parent dgram (with new extent) to the begining of outbuf.
        self.uiter.copy_parent(self.pydg)

        # Release PyBuffer object
        self.uiter.free_outbuf()

    def updatetimestamp(self, timestamp_val):
        self.uiter.updatetimestamp(self.pydg, timestamp_val)

    @property
    def size(self):
        return self.uiter.get_size()

    @property
    def uiter(self):
        return self.uiter

