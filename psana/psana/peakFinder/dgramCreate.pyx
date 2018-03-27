# Import the Python-level symbols of numpy
import numpy as np

# Import the C-level symbols of numpy
cimport numpy as cnp

from cpython cimport array
import array
import sys
import ctypes

from libc.stdlib cimport malloc,free
from cpython cimport PyObject, Py_INCREF
from libc.string cimport memcpy

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
        const char* name()
        DataType    type()
        cnp.uint32_t    rank()

    cdef cppclass Shape:
        Shape(unsigned shape[5])
        cnp.uint32_t* shape()


cdef extern from "xtcdata/xtc/ShapesData.hh" namespace "XtcData":
    cdef cppclass AutoParentAlloc:
        AutoParentAlloc(TypeId typeId): Xtc(typeId)
        # void* alloc(cnp.uint32_t, Xtc& parent)
        # void* alloc(cnp.uint32_t, Xtc& parent, Xtc& superparent)

    cdef cppclass Names(AutoParentAlloc):
        Names(const char* detName, Alg& alg, const char* detType, const char* detId, unsigned segment):\
             # AutoParentAlloc(TypeId(<TypeId.Type>Names,0)), _alg(alg),_segment(segment)
            pass
    cdef cppclass NameIndex: 
        NameIndex(Names &names)

    cdef cppclass Shapes:
        Shapes(Xtc superparent, cnp.uint32_t namesId): AutoParentAlloc(TypeId(<TypeId.Type>Shapes,0)),_namesId(namesId)



# cdef extern from "xtcdata/xtc/src/xtcwriter.cc" namespace "XtcData":
    # void add_names(Xtc parent, vector[NameIndex] namesVec) 

cdef class PyXtc:
    cdef Xtc* cptr
    def __cinit__(self, int sizet, charp):
        self.cptr = new Xtc()

    def __dealloc__(self):
        del self.cptr

cdef class PyShapes:
    cdef Shapes* cptr
    def __cinit__(self, superparent, int namesId):
        self.cptr = new Shapes(<Xtc&>superparent, <cnp.uint32_t>namesId)

    def __dealloc__(self):
        del self.cptr


cdef class PyNames:
    cdef Names* cptr

    def __cinit__(self, detName, alg, detType,detId, segment):
        pass
        # self.cptr = new Names(<const char*>detName, <Alg&>alg, <const char*>detType, <const char*>detId, <unsigned>segment)
        # self.cptr = new Names(detName, <Alg&>alg, detType, detId, segment)

    def __dealloc__(self):
        del self.cptr


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
cdef class PyShape:
    cdef Shape* cptr
    
    def __cinit__(self, cnp.ndarray[cnp.uint32_t, ndim=1, mode="c"] shape):
       self.cptr= new Shape(<cnp.uint32_t*>shape.data)

    def getshape(self, num):
        cdef cnp.uint32_t* shapen = self.cptr.shape()
        return shapen[num]
    def __dealloc__(self):
        del self.cptr


def blockcreate(data):
    cdef Name* namesptr = <Name*> malloc(len(data)*sizeof(Name))
    cdef Shape* shapesptr = <Shape*> malloc(len(data)*sizeof(Shape))
    cdef cnp.uint8_t* dataptr  = <cnp.uint8_t*> malloc(0x4000000)

    cdef cnp.uint8_t* dataptr2  = <cnp.uint8_t*> malloc(0x0100000)

    cdef cnp.ndarray[int, mode="c", ndim=2] shop

    for ct,entry in enumerate(data):
        # Create the name object
        pyname = PyName(entry[0][0],PyAlg(entry[0][1].algname,\
                                              entry[0][1].major, entry[0][1].minor, entry[0][1].micro))
        # Copy the name to namesptr
        memcpy(namesptr, pyname.cptr,sizeof(Name))
        print("Block %i:" % ct)
        print("\tName from py object: %s" % <char*>namesptr.name())
        print("\tName from malloc'd memory: %s" %<char*>pyname.cptr.name())
        namesptr+=1

        # Deduce the shape of the data
        array_size = np.array(entry[1].shape)
        array_rank = len(array_size)
        array_size = np.array(np.r_[array_size, (5-array_rank)*[0]], dtype=np.uint32)
        # Create the shape object
        pyshape = PyShape(array_size)

        # Copy the shape to shapesptr
        memcpy(shapesptr, pyshape.cptr, sizeof(Shape))

        dims = ''
        for i in range(array_rank):
            dims+=str(pyshape.getshape(i))+'x'

        dimsmal = ''
        for i in range(array_rank):
            dimsmal+=str(shapesptr.shape()[i])+'x'

        print("\tShape from py object: %s" % dims[:-1])
        print("\tShape from malloc'd memory: %s" % dimsmal[:-1])

        shapesptr+=1

        data_char = entry[1].ctypes.data_as(ctypes.POINTER(ctypes.c_char))

        for i in range(entry[1].nbytes):
            dataptr2[i] = ord(data_char[i]) 

        memcpy(dataptr, dataptr2, entry[1].nbytes)#  <--- TypeError here

        result_malloc = []
        result_pyobj = []

        for i in range(int(entry[1].nbytes/4)):
            intap = int.from_bytes(dataptr[i*4:(i+1)*4], byteorder='little')
            result_malloc.append(intap)

        for i in range(int(entry[1].nbytes/4)):
            intap = int.from_bytes(data_char[i*4:(i+1)*4], byteorder='little')
            result_pyobj.append(intap)

        res1 = np.array(result_malloc)
        res2 = np.array(result_pyobj)

        print("\tPy obj and malloc'd memory difference %i\n: " %np.sum(res1-res2))
        dataptr+=entry[1].nbytes

        
        # Copy the data to the dataptr buffer
        # arr = array_contig(entry[1])
        # if array_rank == 2:
        #     arr2 = entry[1]
        #     memcpy(dataptr, <cnp.uint8_t*>arr2.data, entry[1].nbytes)#  <--- TypeError here
        # if array_rank == 3:
        #     arr3 = entry[1]
        #     memcpy(dataptr, <cnp.uint8_t*>arr3.data, entry[1].nbytes)#  <--- TypeError here
        # if array_rank == 4:
        #     arr4 = entry[1]
        #     memcpy(dataptr, <cnp.uint8_t*>arr4.data, entry[1].nbytes)#  <--- TypeError here
        # if array_rank == 5:
        #     arr5 = entry[1]
        #     memcpy(dataptr, <cnp.uint8_t*>arr5.data, entry[1].nbytes)#  <--- TypeError here

        # arr = entry[1]
        # dataptr2 = entry[1].ctypes.data_as(ctypes.POINTER(ctypes.c_char))
        #dataptr2 = <cnp.uint8_t*>entry[1].ctypes.data
        # memcpy(dataptr, <cnp.uint8_t*>arr.data, entry[1].nbytes)#  <--- TypeError here
       # memcpy(dataptr, dataptr2, entry[1].nbytes)#  <--- TypeError here
        # dataptr += sizeof(entry[1].data)

     # for _ in range(len(data)):

    # def pyadd_names(Xtc parent, vector[NameIndauex] namesVec):
#    return add_names(parent, namesVec) #cdef Alg* = new Alg("hsd", 0, 0, 0)
# # Alg = PyAlg()
#     # frontEndNames = PyNames("xpphsd", Alg, "foo", "bar",0)

# cdef class CreateDgram:

#     cdef Dgram* cptr

#     def __cinit__(self):
#         cdef int bufsize = 0x4000000
#         cdef int dam = 0
#         cdef void* configbuf = malloc(bufsize)
#         cdef Dgram *config = <Dgram*>configbuf
#         tid = PyTypeId(0,0)
#         dmg = PyDamage(0)

#         config.xtc.contains = tid.cptr[0]
#         config.xtc.damage = dmg.cptr[0]
#         config.xtc.extent = sizeof(Xtc)

#         namesVec = new vector[NameIndex]()
#         # pyadd_names(config.xtc,namesVec)


#     def __dealloc__(self):
#         del self.cptr
