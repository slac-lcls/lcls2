# Import the Python-level symbols of numpy
import numpy as np

# Import the C-level symbols of numpy
cimport numpy as cnp

from cpython cimport array
# import array

from libc.stdlib cimport malloc,free
from cpython cimport PyObject, Py_INCREF
from libc.string cimport memcpy
from libc.stdio cimport *

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
cnp.import_array()

cimport libc.stdint as si

# python classes to make the test script tidier
class alg:
    def __init__(self, name,version):
        self.algname = name
        [self.major,self.minor,self.micro] = version

class nameinfo:
    def __init__(self, dname, detype, did, namesid):
        self.detName = dname
        self.detType = detype
        self.detId = did
        self.namesId = namesid

cdef extern from 'xtcdata/xtc/ShapesData.hh' namespace "XtcData":
    cdef cppclass pyDgram:
        pyDgram(cnp.uint8_t *name_block, cnp.uint8_t* shape_block, size_t block_elems,
                   cnp.uint8_t* data_block, size_t sizeofdata,void* buffdgram)

        void addEvent(cnp.uint8_t* shape_block, size_t block_elems,
                cnp.uint8_t* data_block, size_t sizeofdata,void* buffdgram)

        cnp.uint32_t dgramSize()

# These classes are imported from ShapesData.hh by DescData.hh
cdef extern from "xtcdata/xtc/DescData.hh" namespace "XtcData":
    cdef cppclass AlgVersion:
        AlgVersion(cnp.uint8_t major, cnp.uint8_t minor, cnp.uint8_t micro) except+
        unsigned major()

    cdef cppclass Alg:
        Alg(const char* alg, cnp.uint8_t major, cnp.uint8_t minor, cnp.uint8_t micro)
        const char* name()

    cdef cppclass NameInfo: 
        NameInfo(const char* detname, Alg& alg0, const char* dettype, const char* detid, cnp.uint32_t segment0):alg(alg0), segment(segment0)

    cdef cppclass Name:
        enum DataType: UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64, FLOAT, DOUBLE
        enum MaxRank: maxrank=5
        Name(const char* name, Alg& alg)
        Name(const char* name, DataType type, int rank, Alg& alg)
        const char* name()
        DataType    type()
        cnp.uint32_t    rank()

    cdef cppclass Shape:
        Shape(unsigned shape[5])
        cnp.uint32_t* shape()

cdef class PyDgram:
    cdef pyDgram* cptr
    cdef size_t buffer_size
    cdef block_elems
    cdef cnp.uint8_t* buffer
    cdef cnp.uint8_t* addBuff

    def __cinit__(self, PyNameBlock pyn, PyShapeBlock pys, PyDataBlock pyd):
        if pyn.ct != pys.ct:
            raise AttributeError("Names and shapes blocks have different sizes")
        else:
            self.block_elems = pyn.ct

        # print("size of shape %i" % sizeof(Shape))
        # print("total size is %i" % self.total_size)
        # print("Data asize is %i" % pyd.get_bytes())
        self.buffer_size =0x4000000

        self.buffer  = <cnp.uint8_t*> malloc(self.buffer_size)

        # print("Buffer hash before: %s" % hashlib.md5(buffer).hexdigest())
        self.cptr = new pyDgram(pyn.cptr_start, pys.cptr_start, self.block_elems, pyd.cptr_start, pyd.get_bytes(), self.buffer)
        # print("Buffer hash: %s" % hashlib.md5(self.buffer[:self.cptr.dgramSize()]).hexdigest())

    def addEvent(self, PyShapeBlock pys, PyDataBlock pyd):
        self.cptr.addEvent(pys.cptr_start, self.block_elems, pyd.cptr_start, pyd.get_bytes(), self.buffer)

    def write(self):
        cdef FILE *f = fopen('data.xtc', 'w')
        fwrite(self.buffer, sizeof(cnp.uint8_t), self.cptr.dgramSize(), f)
        fclose(f)

    def __dealloc__(self):
        del self.cptr


cdef class PyAlgVer:
    cdef AlgVersion* cptr
    def __cinit__(self):
        self.cptr = new AlgVersion(9,8,7)

    def __dealloc__(self):
        del self.cptr

    def major(self):
        return self.cptr.major()

cdef class PyAlg:
    cdef Alg* cptr
    def __cinit__(self, bytes algname, major, minor, micro):
        self.cptr = new Alg(algname, major, minor, micro)

    def __dealloc__(self):
        del self.cptr

    def name(self):
        return self.cptr.name()

cdef class PyNameInfo:
    cdef NameInfo* cptr

    def __cinit__(self,detNames, PyAlg pyalg, detType, detId, segment=0):
        self.cptr = new NameInfo(detNames, pyalg.cptr[0], detType, detId, segment)

    def __dealloc__(self):
        del self.cptr

cdef class PyNameBlock:
    cdef cnp.uint8_t* cptr
    cdef cnp.uint8_t* cptr_start
    cdef ct

    def __cinit__(self, num_elems):
        self.cptr = <cnp.uint8_t*>malloc(sizeof(NameInfo)+num_elems*sizeof(Name))
        self.cptr_start = self.cptr
        self.ct = 0

    def addNameInfo(self, PyNameInfo pynameinfo):
        memcpy(self.cptr, pynameinfo.cptr, sizeof(NameInfo))
        self.cptr += sizeof(NameInfo)

    def addName(self, bytes namestring, int type, int rank, PyAlg pyalg):
        newName = new Name(namestring,<Name.DataType>type, rank, pyalg.cptr[0])
        memcpy(self.cptr, newName, sizeof(Name))
        self.cptr+=sizeof(Name)
        self.ct +=1

        # def __dealloc__(self):
    #     del self.cptr

cdef class PyShapeBlock:
    cdef cnp.uint8_t* cptr
    cdef cnp.uint8_t* cptr_start
    cdef ct

    def __cinit__(self, cnp.uint32_t namesId, int num_elems):
        self.cptr = <cnp.uint8_t*> malloc(sizeof(cnp.uint32_t)+num_elems*sizeof(Shape))
        self.cptr_start = self.cptr
        memcpy(self.cptr, &namesId, sizeof(cnp.uint32_t))
        self.cptr += sizeof(cnp.uint32_t)
        self.ct = 0
    def addShape(self,  cnp.ndarray[cnp.uint32_t, ndim=1, mode="c"] shape):
        newShape = new Shape(<cnp.uint32_t*>shape.data)
        memcpy(self.cptr, newShape, sizeof(Shape))
        self.cptr += sizeof(Shape)
        self.ct +=1
    # def __dealloc__(self):
    #     #del self.cptr

cdef class PyDataBlock:
    cdef cnp.uint8_t* cptr
    cdef cnp.uint8_t* cptr_start

    def __cinit__(self):
        self.cptr = <cnp.uint8_t*> malloc(0x4000000)
        self.cptr_start = self.cptr

    def addData(self, array):
         cdef cnp.ndarray[cnp.uint8_t, ndim=1, mode='c'] carr
         carr = np.array(bytearray(array.tobytes()), dtype=np.uint8)
         memcpy(self.cptr, carr.data, carr.nbytes)
         self.cptr+=carr.nbytes

    def get_bytes(self):
        return int(self.cptr - self.cptr_start)

def parse_type(data_arr):
    dat_type = data_arr.dtype

    types = ['uint8','uint16','uint32','uint64','int8','int16','int32','int64','float','double']
    nptypes = list(map(lambda x: np.dtype(x), types))
    type_dict = dict(zip(nptypes, range(len(nptypes))))
    return type_dict[dat_type]


def blockcreate(data, verbose=False):

    # detName = data[0][0]
    # detType = data[0][1]
    # detId = data[0][2]
    # namesId = data[0][3]

    py_shape = PyShapeBlock(data[0][0].namesId, len(data[1:]))
    py_name = PyNameBlock(len(data[1:]))
    py_data = PyDataBlock()

    # Create the alg object
    pyalg = PyAlg(data[1][0][1].algname,data[1][0][1].major, data[1][0][1].minor, data[1][0][1].micro)

    py_nameinfo = PyNameInfo(data[0][0].detName, pyalg, data[0][0].detType, data[0][0].detId)
    py_name.addNameInfo(py_nameinfo)

    for ct,entry in enumerate(data[1:]):

        # Deduce the shape of the data
        array_size = np.array(entry[1].shape)
        array_rank = len(array_size)
        array_size_pad = np.array(np.r_[array_size, (5-array_rank)*[0]], dtype=np.uint32)
        # Find the type of the data
        data_type = parse_type(entry[1])

        # Create the alg
        pyalg = PyAlg(entry[0][1].algname,entry[0][1].major, entry[0][1].minor, entry[0][1].micro)

        # Copy the name to the block
        py_name.addName(entry[0][0],data_type, array_rank,pyalg)

        # Copy the shape to the block
        py_shape.addShape(array_size_pad)

        carr = np.array(bytearray(entry[1].tobytes()), dtype=np.uint8)
        py_data.addData(entry[1])

  
    pydgram = PyDgram(py_name, py_shape, py_data)
    pydgram.addEvent(py_shape, py_data)
    pydgram.write()


