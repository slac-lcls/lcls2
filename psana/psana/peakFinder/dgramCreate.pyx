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

def fix_encoding(input):
    try:
        return input.encode()
    except AttributeError:
        return input

class alg:
    def __init__(self, name,version):
        # cast to bytestring
        self.algname = fix_encoding(name)
        [self.major,self.minor,self.micro] = version

class nameinfo:
    def __init__(self, dname, detype, did, namesid):
        self.detName = fix_encoding(dname)
        self.detType = fix_encoding(detype)
        self.detId = fix_encoding(did)
        self.namesId = namesid

class parse_xtc():
    def __init__(self, datasource):
        self.datasource = datasource
        self.events_dict = []
        self.config_dict = {}
        self.parse_configure()

    def parse_configure(self):
        config = vars(self.datasource.configs[0])
        sw_config = vars(config['software'])
        det_names = [x for x in config.keys() if x not in ('software')]

        det_dict = {}
        namesid = 0
        for detector in det_names:
            det_sw_config = vars(sw_config[detector])
            det_config = vars(config[detector])


            alg_types = [x for x in det_config.keys() if x not in ('detid', 'dettype')]
            det_entries = []
            for algt in alg_types:
                ninfo = nameinfo(detector, det_sw_config['dettype'], \
                                    det_sw_config['detid'], namesid)
                namesid += 1
                base_alg = alg(det_sw_config[algt].software, det_sw_config[algt].version)
                base_alg_name = det_sw_config[algt].software
                data_algs_block = vars(det_sw_config[algt])
                data_algs = {}
                data = vars(det_config[algt])
                for data_name in data.keys():
                    alg_dict = data_algs_block[data_name]
                    minor_alg = alg(alg_dict.software, alg_dict.version)
                    data_algs[data_name] = [data[data_name], minor_alg]

                det_entries.append({'nameinfo':ninfo, 'base_alg':base_alg, \
                                    'base_alg_name':base_alg_name, 'data':data_algs})

            det_dict[detector] = det_entries

        self.config_dict=det_dict
        self.events_dict.append(det_dict)

    def parse_event(self, evt):
        event_dict = {}
        for key, value in self.config_dict.items():
            det_entries = []
            for iterd in value:
                self.iterd = iterd
                try:
                    dgram = vars(vars(evt.dgrams[0])[key])
                except KeyError:
                    continue
                event_data = vars(dgram[iterd['base_alg_name']])
                det_entries.append({'nameinfo':iterd['nameinfo'], 'base_alg':iterd['base_alg'],\
                                    'data':event_data})
            if det_entries:
                event_dict[key] = det_entries

        self.events_dict.append(event_dict)

    def write_events(self, fileName, cydgram):
        with open(fileName, 'wb') as f:
            for event in self.events_dict:
                for key, value in event.items():
                    for detector in value:
                        cydgram.addDet(detector['nameinfo'], detector['base_alg'], detector['data'])
                xtc_byte = cydgram.get()
                f.write(xtc_byte)

            # pydgram.writeToFile()



cdef extern from 'xtcdata/xtc/ShapesData.hh' namespace "XtcData":

    cdef cppclass blockDgram:
        blockDgram(void* buffdgram)

        void addNamesBlock(cnp.uint8_t* name_block, size_t block_elems)
        void addShapesDataBlock(cnp.uint8_t* shape_block, cnp.uint8_t* data_block,\
                           size_t sizeofdata, size_t block_elems)

        void addDataBlock(cnp.uint8_t* data_block,size_t sizeofdata)


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
        NameInfo(const char* detname, Alg& alg0, const char* dettype, const char* detid, cnp.uint32_t segment0, cnp.uint32_t numarr):alg(alg0), segment(segment0)

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

cdef class PyBlockDgram:
    cdef blockDgram* cptr
    cdef size_t buffer_size
    cdef cnp.uint8_t* buffer
    cdef cnp.uint8_t* addBuff
    cdef size_t block_elems


    def __cinit__(self):

        self.buffer_size =0x4000000
        self.buffer  = <cnp.uint8_t*> malloc(self.buffer_size)
        self.cptr = new blockDgram(self.buffer)

    def addNamesBlock(self, PyNameBlock pyn):
        self.cptr.addNamesBlock(pyn.cptr_start, pyn.ct)

    def addShapesDataBlock(self, PyShapeBlock pys, PyDataBlock pyd):
        if pys.ct>0:
            self.cptr.addShapesDataBlock(pys.cptr_start, pyd.cptr_start, pyd.get_bytes(), pys.ct)
        else:
            self.cptr.addDataBlock(pyd.cptr_start, pyd.get_bytes())

    def writeToFile(self, filename):
        cdef FILE *f = fopen(filename, 'a')
        fwrite(self.buffer, sizeof(cnp.uint8_t), self.cptr.dgramSize(), f)
        fclose(f)

    def retByArr(self):
        cdef cnp.uint8_t[:] view = <cnp.uint8_t[:self.cptr.dgramSize()]>self.buffer
        numpy_arr = np.asarray(view)
        return numpy_arr

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

    def __cinit__(self,detNames, PyAlg pyalg, detType, detId, numarr, segment=0):
        self.cptr = new NameInfo(detNames, pyalg.cptr[0], detType, detId, segment, numarr)

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
        self.cptr += sizeof(NameInfo)

    def addNameInfo(self, PyNameInfo pynameinfo):
        memcpy(self.cptr_start, pynameinfo.cptr, sizeof(NameInfo))

    def addName(self, bytes namestring, PyAlg alg, int type=-1, int rank=-1):
        if type == -1 and rank == -1:
            newName = new Name(namestring, alg.cptr[0])
        else:
            newName = new Name(namestring,<Name.DataType>type, rank, alg.cptr[0])
        memcpy(self.cptr, newName, sizeof(Name))
        self.cptr+=sizeof(Name)
        self.ct +=1

    def __dealloc__(self):
        free(self.cptr_start)


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

    def __dealloc__(self):
        free(self.cptr_start)

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

    def __dealloc__(self):
        free(self.cptr_start)


def parse_type(data_arr):
    dat_type = data_arr.dtype
    types = ['uint8','uint16','uint32','uint64','int8','int16','int32','int64','float32','float64']
    nptypes = list(map(lambda x: np.dtype(x), types))
    type_dict = dict(zip(nptypes, range(len(nptypes))))
    return type_dict[dat_type]
 
class CyDgram():
    def __init__(self):
        self.write_configure = True
        # self.filename = fix_encoding(filename)
        self.config_block = []

    def addDet(self, nameinfo, alg, event_dict):
        num_elem = len(event_dict)
        py_shape = PyShapeBlock(nameinfo.namesId, num_elem)
        py_name = PyNameBlock(num_elem)
        py_data = PyDataBlock()

        basealg = PyAlg(alg.algname,alg.major, alg.minor, alg.micro)

        num_arrays = 0
        for name, array in event_dict.items():
            try:
                array_alg = bool(type(array[1]) == type(alg))
            except (IndexError,TypeError):
                array_alg = False
            if array_alg:
                arr = np.asarray(array[0])
                pyalg = PyAlg(array[1].algname,array[1].major, array[1].minor, array[1].micro)
            else:
                arr = np.asarray(array)
                pyalg = basealg

            # Deduce the shape of the data
            array_size = np.array(arr.shape)
            array_rank = len(array_size)

            array_size_pad = np.array(np.r_[array_size, (5-array_rank)*[0]], dtype=np.uint32)
            # Find the type of the data
            data_type = parse_type(arr)
            # Copy the name to the block
            py_name.addName(fix_encoding(name), pyalg, data_type, array_rank)

            # Copy the shape to the block
            if array_rank > 0:
                num_arrays += 1
                py_shape.addShape(array_size_pad)
            py_data.addData(arr)

        py_nameinfo = PyNameInfo(nameinfo.detName, basealg, nameinfo.detType, nameinfo.detId, num_arrays)
        py_name.addNameInfo(py_nameinfo)
        self.config_block.append([py_name, py_shape, py_data])

    def constructBlock(self):
        self.pydgram = PyBlockDgram()

        if self.write_configure:
            for name, _, _ in self.config_block:
                self.pydgram.addNamesBlock(name)
            self.write_configure = False

        for _, shape, data in self.config_block:
            self.pydgram.addShapesDataBlock(shape, data)
     # def writeToFile(self):
     #    if self.config_block:
     #        self.constructBlock()
     #        self.config_block = []
     #    self.pydgram.writeToFile(self.filename)

    def get(self):
        self.constructBlock()
        self.config_block = []
        return self.pydgram.retByArr().tobytes()

    def getArray(self):
        self.constructBlock()
        self.config_block = []
        return self.pydgram.retByArr()

