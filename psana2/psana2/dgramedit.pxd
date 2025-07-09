cimport numpy as cnp
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t

cnp.import_array()  # needed

cdef extern from 'xtcdata/xtc/TransitionId.hh' namespace "XtcData":
    cdef cppclass TransitionId:
        enum Value: ClearReadout, Reset, Configure, Unconfigure, BeginRun, EndRun, \
            BeginStep, EndStep, Enable, Disable, SlowUpdate, L1Accept_EndOfBatch, L1Accept = 12, NumberOf

cdef extern from 'xtcdata/xtc/TimeStamp.hh' namespace "XtcData":
    cdef cppclass TimeStamp:
        TimeStamp() except +
        uint64_t value()

cdef extern from 'xtcdata/xtc/Dgram.hh' namespace "XtcData":
    cdef cppclass Dgram:
        Dgram() except +
        Xtc xtc
        TransitionId.Value service()
        TimeStamp time

cdef extern from "xtcdata/xtc/Damage.hh" namespace "XtcData":
    cdef cppclass Damage:
        Damage() except +
        uint16_t value() const

cdef extern from 'xtcdata/xtc/Xtc.hh' namespace "XtcData":
    cdef cppclass Xtc:
        Xtc() except +
        int sizeofPayload() const
        uint32_t extent
        Damage damage

cdef extern from "xtcdata/xtc/NamesId.hh" namespace "XtcData":
    cdef cppclass NamesId:
        NamesId() except +

cdef extern from "xtcdata/xtc/ShapesData.hh" namespace "XtcData":
    cdef cppclass ShapesData:
        ShapesData(NamesId& namesId) except +

cdef extern from "xtcdata/xtc/NameIndex.hh" namespace "XtcData":
    cdef cppclass NameIndex:
        NameIndex() except +

cdef extern from "xtcdata/xtc/DescData.hh" namespace "XtcData":
    cdef cppclass DescData:
        DescData(ShapesData& shapesdata, NameIndex& nameindex) except +

    cdef cppclass AlgVersion:
        AlgVersion(cnp.uint8_t major, cnp.uint8_t minor, cnp.uint8_t micro) except+
        unsigned major()

    cdef cppclass Alg:
        Alg(const char* alg, cnp.uint8_t major, cnp.uint8_t minor, cnp.uint8_t micro)
        const char* name()

    cdef cppclass NameInfo:
        NameInfo(const char* detname, Alg& alg0, const char* dettype,
                 const char* detid, cnp.uint32_t segment0, cnp.uint32_t numarr):
            alg(alg0), segment(segment0)

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

cdef extern from 'xtcdata/xtc/XtcFileIterator.hh' namespace "XtcData":
    cdef cppclass XtcFileIterator:
        XtcFileIterator(int fd, size_t maxDgramSize) except +
        Dgram* next()
        uint64_t size()


cdef extern from 'xtcdata/xtc/XtcUpdateIter.hh' namespace "XtcData":

    cdef cppclass XtcUpdateIter:
        XtcUpdateIter(unsigned numWords) except +
        int process(Xtc* xtc, const void* bufEnd)
        void get_value(int i, Name& name, DescData& descdata)
        void iterate(Xtc* xtc, const void* bufEnd)
        unsigned getSize()
        unsigned getNodeId()
        unsigned getNextNamesId()
        void addNames(Xtc& xtc, const void* bufEnd, char* detName, char* detType, char* detId,
                      unsigned nodeId, unsigned namesId, unsigned segment,
                      char* algName, uint8_t major, uint8_t minor, uint8_t micro,
                      DataDef& datadef)
        void setString(char* data, DataDef& datadef, char* varname)
        void setValue(unsigned nodeId, unsigned namesId,
                      char* data, DataDef& datadef, char* varname)
        void setOutput(char* outbuf)
        void addData(unsigned nodeId, unsigned namesId,
                     unsigned* shape, char* data, DataDef& datadef, char* varname)
        Dgram& createTransition(unsigned transId, unsigned counting_timestamps,
                                uint64_t timestamp_val, char* buf)
        void createData(Xtc& xtc, const void* bufEnd, unsigned nodeId, unsigned namesId)
        void updateTimeStamp(Dgram& d, uint64_t timestamp_val)
        void updateService(Dgram& d, uint8_t transitionId)
        void updateDamage(Dgram& d, uint16_t damage)
        int  getElementSize(unsigned nodeId, unsigned namesId,
                            DataDef& datadef, char* varname)
        cnp.uint32_t getRemovedSize()
        void copyParent(Dgram* parent_d)
        void setFilter(char* detName, char* algName)
        void setCfgFlag(int cfgFlag)
        void setCfgWriteFlag(int cfgWriteFalg)
        int isConfig()

    cdef cppclass DataDef:
        DataDef() except +
        void show()
        void add(char* name, unsigned dtype, int rank)
        int getDtype(char* name)
        int getRank(char* name)
