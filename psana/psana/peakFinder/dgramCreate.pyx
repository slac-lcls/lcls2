# Import the Python-level symbols of numpy
import numpy as np

# Import the C-level symbols of numpy
cimport numpy as cnp

from cpython cimport array
import array
import sys
import ctypes
import hashlib
import copy

from libc.stdlib cimport malloc,free
from cpython cimport PyObject, Py_INCREF
from libc.string cimport memcpy
from libc.stdio cimport *

# C++ vector class
from libcpp.vector cimport vector
# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
cnp.import_array()

from libcpp.vector cimport vector
cimport libc.stdint as si


cdef extern from "xtcdata/xtc/Xtc.hh" namespace "XtcData":
    cdef cppclass Xtc:
        Xtc(...) : damage(0), extent(0)
        Xtc2 "Xtc"(const Xtc& xtc): damage(xtc.damage), src(xtc.src), contains(xtc.contains), extent(sizeof(Xtc))
        Xtc3 "Xtc"(const TypeId& type) : damage(0), contains(type), extent(sizeof(Xtc))
        Damage   damage
        Src      src
        TypeId   contains
        cnp.uint32_t extent


cdef extern from "xtcdata/xtc/Src.hh" namespace "XtcData":
    cdef cppclass Src:
        Src()
cdef extern from "xtcdata/xtc/Damage.hh" namespace "XtcData":
    cdef cppclass Damage:
        Damage(int v)

cdef extern from "xtcdata/xtc/Dgram.hh" namespace "XtcData":
    cdef cppclass Dgram:
        Dgram()
        Xtc xtc


cdef extern from "xtcdata/xtc/TypeId.hh" namespace "XtcData":
    cdef cppclass TypeId:
        enum Type: Parent, ShapesData, Shapes, Data, Names, NumberOf
        TypeId(Type type, int version)


# These classes are imported from ShapesData.hh by DescData.hh
cdef extern from "xtcdata/xtc/DescData.hh" namespace "XtcData":
    cdef cppclass AlgVersion:
        AlgVersion(cnp.uint8_t major, cnp.uint8_t minor, cnp.uint8_t micro) except+
        unsigned major()

    cdef cppclass Alg:
        Alg(const char* alg, cnp.uint8_t major, cnp.uint8_t minor, cnp.uint8_t micro)
        const char* name()


    cdef cppclass Name:
        enum DataType: UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64, FLOAT, DOUBLE
        enum MaxRank: maxrank=5
        # Name(const char* name, DataType type, int rank)
        # Name2 "Name"(const char* name, DataType type, int rank,Alg& alg)
        Name(const char* name, Alg& alg)
        Name(const char* name, DataType type, int rank, Alg& alg)
        const char* name()
        DataType    type()
        cnp.uint32_t    rank()

    cdef cppclass Shape:
        Shape(unsigned shape[5])
        cnp.uint32_t* shape()


cdef class PyTypeId:
    cdef TypeId* cptr

    def __cinit__(self, int type, int version):
        self.cptr = new TypeId(<TypeId.Type>type,version)

    def __dealloc__(self):
        del self.cptr

cdef class PyDamage:
    cdef Damage* cptr

    def __cinit__(self, int damage):
       self.cptr = new Damage(damage)
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

cdef class PyName:
    cdef Name* cptr

    def __cinit__(self, bytes namestring, PyAlg pyalg):
        # self.cptr = new Name(namestring,<Name.DataType>type,version)
        self.cptr = new Name(namestring,pyalg.cptr[0])

    def __dealloc__(self):
        del self.cptr

    def type(self):
        return self.cptr.type()


    def name(self):
         return <char*>self.cptr.name()

cdef class PyName2:
    cdef Name* cptr

    def __cinit__(self, bytes namestring, int type, int rank, PyAlg pyalg):
        self.cptr = new Name(namestring,<Name.DataType>type, rank, pyalg.cptr[0])

    def __dealloc__(self):
        del self.cptr

    def type(self):
        return self.cptr.type()


    def name(self):
         return <char*>self.cptr.name()


cdef class PyShape:
    cdef Shape* cptr
    
    def __cinit__(self, cnp.ndarray[cnp.uint32_t, ndim=1, mode="c"] shape):
       self.cptr= new Shape(<cnp.uint32_t*>shape.data)

    def getshape(self, num):
        cdef cnp.uint32_t* shapen = self.cptr.shape()
        return shapen[num]
    def __dealloc__(self):
        del self.cptr

def parse_type(data_arr):
    dat_type = data_arr.dtype

    types = ['uint8','uint16','uint32','uint64','int8','int16','int32','int64','float','double']
    nptypes = list(map(lambda x: np.dtype(x), types))
    type_dict = dict(zip(nptypes, range(len(nptypes))))
    return type_dict[dat_type]

def blockcreate(data, verbose=False):
    cdef Name* namesptr = <Name*> malloc(len(data)*sizeof(Name))
    cdef Shape* shapesptr = <Shape*> malloc(len(data)*sizeof(Shape))
    cdef cnp.uint8_t* dataptr  = <cnp.uint8_t*> malloc(0x4000000)
    cdef cnp.uint8_t* dataptr2  = <cnp.uint8_t*> malloc(0x0100000)

    cdef cnp.ndarray[int, mode="c", ndim=2] shop
    xtc_bytes = 0
    for ct,entry in enumerate(data):
        # Deduce the shape of the data
        array_size = np.array(entry[1].shape)
        array_rank = len(array_size)
        array_size_pad = np.array(np.r_[array_size, (5-array_rank)*[0]], dtype=np.uint32)
        # Find the type of the data
        data_type = parse_type(entry[1])

        # Create the name object
        pyname = PyName2(entry[0][0],data_type, array_rank, PyAlg(entry[0][1].algname,\
                                          entry[0][1].major, entry[0][1].minor, entry[0][1].micro))
        # Copy the name to namesptr
        memcpy(namesptr, pyname.cptr,sizeof(Name))
        if verbose:
            print("Block %i:" % ct)
            print("\tName from py object: %s" % <char*>namesptr.name())
            print("\tName from malloc'd memory: %s" %<char*>pyname.cptr.name())
        namesptr+=1
        # Create the shape object
        pyshape = PyShape(array_size_pad)

        # Copy the shape to shapesptr
        memcpy(shapesptr, pyshape.cptr, sizeof(Shape))

        dims = ''
        for i in range(array_rank):
            dims+=str(pyshape.getshape(i))+'x'

        dimsmal = ''
        for i in range(array_rank):
            dimsmal+=str(shapesptr.shape()[i])+'x'
        if verbose:
            print("\tShape from py object: %s" % dims[:-1])
            print("\tShape from malloc'd memory: %s" % dimsmal[:-1])

        shapesptr+=1

        data_char = entry[1].ctypes.data_as(ctypes.POINTER(ctypes.c_char))

        for i in range(entry[1].nbytes):
            dataptr2[i] = ord(data_char[i]) 

        memcpy(dataptr, dataptr2, entry[1].nbytes)#  <--- TypeError here

        result_malloc = []
        result_pyobj = []

        bstr_m = dataptr[0:entry[1].nbytes]
        bstr_py = data_char[0:entry[1].nbytes]

        data_m = np.resize(np.frombuffer(bstr_m, dtype = entry[1].dtype), array_size)
        data_py = np.resize(np.frombuffer(bstr_py, dtype = entry[1].dtype), array_size)

        dataptr += entry[1].nbytes
        xtc_bytes += entry[1].nbytes
        if verbose:
            print("malloc'd array hash: ", hashlib.md5(data_m.tobytes()).hexdigest())
            print("py array hash: ", hashlib.md5(data_py.tobytes()).hexdigest())

    dataptr -= xtc_bytes
    shapesptr -= len(data)
    namesptr -= len(data)

    cdef FILE *f = fopen('data.xtc', 'w')
    fwrite(&dataptr, xtc_bytes, 1, f)
    fwrite(&shapesptr, len(data)*sizeof(Shape), 1, f)
    fwrite(&namesptr, len(data)*sizeof(Name), 1, f)
    fclose(f)

