from psana.dgramlite cimport Xtc, Dgram
from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t


cdef extern from *:
    """
    #include <cstddef>
    #include <cstdint>
    #include <cstring>

    #include "xtcdata/xtc/DescData.hh"
    #include "xtcdata/xtc/Dgram.hh"
    #include "xtcdata/xtc/NamesIter.hh"
    #include "xtcdata/xtc/ShapesData.hh"
    #include "xtcdata/xtc/TypeId.hh"
    #include "xtcdata/xtc/XtcIterator.hh"

    namespace PsanaDgramLite {

    class SmdInfoExtractIter : public XtcData::XtcIterator {
    public:
        enum { Stop, Continue };

        SmdInfoExtractIter(XtcData::Xtc* xtc,
                           const void* bufEnd,
                           XtcData::NamesLookup& namesLookup) :
            XtcData::XtcIterator(xtc, bufEnd),
            _namesLookup(namesLookup),
            _offset(0),
            _size(0),
            _found(false)
        {
        }

        int process(XtcData::Xtc* xtc, const void* bufEnd)
        {
            switch (xtc->contains.id()) {
            case XtcData::TypeId::Parent:
                iterate(xtc, bufEnd);
                break;
            case XtcData::TypeId::Names: {
                XtcData::Names& names = *(XtcData::Names*)xtc;
                _namesLookup[names.namesId()] = XtcData::NameIndex(names);
                break;
            }
            case XtcData::TypeId::ShapesData: {
                XtcData::ShapesData& shapesData = *(XtcData::ShapesData*)xtc;
                XtcData::NamesId namesId = shapesData.namesId();
                if (_namesLookup.count(namesId) <= 0) {
                    break;
                }

                XtcData::NameIndex& nameIndex = _namesLookup[namesId];
                XtcData::Names& names = nameIndex.names();
                if (std::strcmp(names.detName(), "smdinfo") != 0 ||
                    std::strcmp(names.alg().name(), "offsetAlg") != 0) {
                    break;
                }

                int offsetIndex = -1;
                int sizeIndex = -1;
                for (unsigned i = 0; i < names.num(); ++i) {
                    XtcData::Name& name = names.get(i);
                    if (std::strcmp(name.name(), "intOffset") == 0) {
                        offsetIndex = i;
                    } else if (std::strcmp(name.name(), "intDgramSize") == 0) {
                        sizeIndex = i;
                    }
                }

                if (offsetIndex >= 0 && sizeIndex >= 0) {
                    XtcData::DescData descData(shapesData, nameIndex);
                    _offset = descData.get_value<uint64_t>(offsetIndex);
                    _size = descData.get_value<uint64_t>(sizeIndex);
                    _found = true;
                    return Stop;
                }
                break;
            }
            default:
                break;
            }
            return Continue;
        }

        bool found() const { return _found; }
        uint64_t offset() const { return _offset; }
        uint64_t size() const { return _size; }

    private:
        XtcData::NamesLookup& _namesLookup;
        uint64_t _offset;
        uint64_t _size;
        bool _found;
    };

    class SmdInfoLiteReader {
    public:
        SmdInfoLiteReader() {}

        int add_names(const void* dgramBuffer, std::size_t nbytes)
        {
            if (dgramBuffer == nullptr || nbytes < sizeof(XtcData::Dgram)) {
                return 0;
            }
            XtcData::Dgram* dgram = (XtcData::Dgram*)dgramBuffer;
            std::size_t dgramSize = sizeof(XtcData::Dgram) + dgram->xtc.sizeofPayload();
            if (dgramSize > nbytes) {
                return 0;
            }
            const void* bufEnd = (const char*)dgram + dgramSize;
            XtcData::NamesIter namesIter(&(dgram->xtc), bufEnd);
            namesIter.iterate();
            _namesLookup = namesIter.namesLookup();
            return (int)_namesLookup.size();
        }

        int extract(const void* dgramBuffer,
                    std::size_t nbytes,
                    uint64_t* offset,
                    uint64_t* size)
        {
            if (offset == nullptr || size == nullptr) {
                return 0;
            }
            *offset = 0;
            *size = 0;

            if (dgramBuffer == nullptr || nbytes < sizeof(XtcData::Dgram)) {
                return 0;
            }
            XtcData::Dgram* dgram = (XtcData::Dgram*)dgramBuffer;
            std::size_t dgramSize = sizeof(XtcData::Dgram) + dgram->xtc.sizeofPayload();
            if (dgramSize > nbytes) {
                return 0;
            }

            const void* bufEnd = (const char*)dgram + dgramSize;
            SmdInfoExtractIter iter(&(dgram->xtc), bufEnd, _namesLookup);
            iter.iterate();
            if (!iter.found()) {
                return 0;
            }

            *offset = iter.offset();
            *size = iter.size();
            return 1;
        }

        int n_names() const { return (int)_namesLookup.size(); }

    private:
        XtcData::NamesLookup _namesLookup;
    };

    } // namespace PsanaDgramLite
    """

    cdef cppclass SmdInfoLiteReader "PsanaDgramLite::SmdInfoLiteReader":
        SmdInfoLiteReader() except +
        int add_names(const void* dgramBuffer, size_t nbytes) except +
        int extract(const void* dgramBuffer, size_t nbytes, uint64_t* offset, uint64_t* size) except +
        int n_names() const

cdef class DgramLite:
    cdef uint64_t payload
    cdef uint64_t timestamp
    cdef unsigned service

    def __init__(self, view):
        """Create Dgram (light-weight version) from a given buffer object."""
        cdef Dgram* d
        cdef char* view_ptr
        cdef Py_buffer buf

        PyObject_GetBuffer(view, &buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
        view_ptr = <char *>buf.buf
        d = <Dgram *>(view_ptr)
        self.payload = d.xtc.extent - sizeof(Xtc)
        self.timestamp = <uint64_t>d.seq.high << 32 | d.seq.low
        self.service = (d.env>>24)&0xf
        PyBuffer_Release(&buf)

    @property
    def payload(self):
        return self.payload

    @property
    def timestamp(self):
        return self.timestamp

    @property
    def service(self):
        return self.service

cdef class SmdInfoLite:
    cdef SmdInfoLiteReader* _reader

    def __cinit__(self):
        self._reader = NULL
        self._reader = new SmdInfoLiteReader()

    def __init__(self, config_view=None, Py_ssize_t offset=0):
        """Create a small SMD-info reader from a Configure dgram buffer."""
        if config_view is not None:
            self.add_names(config_view, offset)

    def __dealloc__(self):
        if self._reader != NULL:
            del self._reader

    def add_names(self, view, Py_ssize_t offset=0):
        """Cache Names from a Configure dgram buffer."""
        cdef Py_buffer buf
        cdef char* view_ptr
        cdef size_t nbytes
        cdef int n_names

        PyObject_GetBuffer(view, &buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
        try:
            if offset < 0 or offset >= buf.len:
                raise ValueError("offset is outside the input buffer")
            view_ptr = <char *>buf.buf + offset
            nbytes = <size_t>(buf.len - offset)
            n_names = self._reader.add_names(<const void*>view_ptr, nbytes)
        finally:
            PyBuffer_Release(&buf)

        return n_names

    def extract(self, view, Py_ssize_t offset=0):
        """Return (found, intOffset, intDgramSize) from an SMD L1 dgram."""
        cdef Py_buffer buf
        cdef char* view_ptr
        cdef size_t nbytes
        cdef uint64_t bd_offset = 0
        cdef uint64_t bd_size = 0
        cdef int found

        PyObject_GetBuffer(view, &buf, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
        try:
            if offset < 0 or offset >= buf.len:
                raise ValueError("offset is outside the input buffer")
            view_ptr = <char *>buf.buf + offset
            nbytes = <size_t>(buf.len - offset)
            found = self._reader.extract(<const void*>view_ptr, nbytes, &bd_offset, &bd_size)
        finally:
            PyBuffer_Release(&buf)

        return bool(found), bd_offset, bd_size

    @property
    def n_names(self):
        return self._reader.n_names()
