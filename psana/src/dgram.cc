
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/TypeId.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/NamesIter.hh"

#include <Python.h>
#include <stdio.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <structmember.h>
#include <assert.h>

#ifdef PSANA_USE_LEGION
#include "legion_helper.h"
#endif

using namespace XtcData;
#define BUFSIZE 0x4000000
#define TMPSTRINGSIZE 256
#define CHUNKSIZE 1<<20
#define MAXRETRIES 5

using namespace std;

// to avoid compiler warnings for debug variables
#define _unused(x) ((void)(x))

struct buffered_reader_t {
public:
    int fd;
    char *chunk;
    size_t offset;
    size_t got;
};

struct PyDgramObject {
    PyObject_HEAD
    PyObject* dict;
    PyObject* pyseq;
    bool is_pyseq_valid;
// Please do not access the dgram field directly. Instead use dgram()
    Dgram* dgram_;
    Dgram*& dgram();
#ifdef PSANA_USE_LEGION
    bool is_dgram_valid;
    LegionArray array;
#endif
    int file_descriptor;
    ssize_t offset;
    buffered_reader_t* reader;
    Py_buffer buf;
};

Dgram*& PyDgramObject::dgram()
{
#ifdef PSANA_USE_LEGION
    if (array && !is_dgram_valid) {
        dgram_ = (Dgram*)array.get_pointer();
        is_dgram_valid = true;
    }
#endif
  return dgram_;
}

/* buffered_reader */
// reads and stores data in a chunk of CHUNKSIZE bytes
// when fails, try to read until MAXRETRIES is reached.

buffered_reader_t *buffered_reader_new(int fd) {
    buffered_reader_t *reader = new buffered_reader_t;
    reader->fd = fd;
    reader->chunk = 0;
    reader->offset = 0;
    reader->got = 0;
    return reader;
}

static void buffered_reader_free(buffered_reader_t *reader) {
    delete reader;
}

static ssize_t read_with_retries(int fd, void *buf, size_t count, int retries) {
    size_t requested = count;
    for (int attempt = 0; attempt < retries; attempt++) {
        size_t got = read(fd, buf, count);
        if (got == count) { // got what we wanted
            return requested;
        } else { // need to wait for more
            buf = (void *)(((char *)buf) + got);
            count -= got;
            // FIXME: sleep?
        }
    }
    return requested - count; // return with whatever got at time out
}

static int buffered_reader_read(buffered_reader_t *reader, void *buf, size_t count) {
    if (!reader->chunk) {
        reader->chunk = (char *)malloc(CHUNKSIZE);
        reader->got = read_with_retries(reader->fd, reader->chunk, CHUNKSIZE, MAXRETRIES);
    }
    
    int read_success = -1;
    while (count > 0 && reader->got > 0) {
        size_t remaining = reader->got - reader->offset;
        if (count < remaining) {
            // just copy it
            memcpy(buf, reader->chunk + reader->offset, count);
            reader->offset += count;
            count = 0;
            read_success = 0;
        } else {
            // copy rest of chunk
            memcpy(buf, reader->chunk + reader->offset, remaining);

            // get new chunk
            reader->got = read_with_retries(reader->fd, reader->chunk, CHUNKSIZE, MAXRETRIES);
            reader->offset = 0;
            buf = (void *)(((char *)buf) + remaining);
            count -= remaining;
        }
    }

    return read_success;
}

static int read_dgram(PyDgramObject* self) {
    // reads dgram header and payload
    int read_header_success = buffered_reader_read(self->reader, self->dgram(), sizeof(Dgram));
    int read_payload_success = buffered_reader_read(self->reader, self->dgram() + 1, self->dgram()->xtc.sizeofPayload());
    return (read_header_success & read_payload_success); 
}

/* end buffered_reader */

static void addObj(PyDgramObject* dgram, const char* name, PyObject* obj) {
    // these three initializations should happen once per event - cpo
    PyObject* containermod = PyImport_ImportModule("psana.container");
    PyObject* pycontainertype = PyObject_GetAttrString(containermod,"Container");
    PyObject* arglist = Py_BuildValue("(O)", dgram);

    char namecopy[TMPSTRINGSIZE];
    strncpy(namecopy,name,TMPSTRINGSIZE);
    PyObject* parent = (PyObject*)dgram;
    char *key = ::strtok(namecopy,"_");
    while(1) {
        char* next = ::strtok(NULL, "_");
        bool last = (next == NULL);
        if (last) {
            // add the real object
            int fail = PyObject_SetAttrString(parent, key, obj);
            if (fail) printf("Dgram: failed to set object attribute\n");
            Py_DECREF(obj); // transfer ownership to parent
            break;
        } else {
            if (!PyObject_HasAttrString(parent, key)) {
                PyObject* container = PyObject_CallObject(pycontainertype, arglist);
                int fail = PyObject_SetAttrString(parent, key, container);
                if (fail) printf("Dgram: failed to set container attribute\n");
                Py_DECREF(container); // transfer ownership to parent
            }
            parent = PyObject_GetAttrString(parent, key);
        }
        key=next;
    }

    Py_DECREF(arglist);
    Py_DECREF(containermod);
    Py_DECREF(pycontainertype);
}

static void setAlg(PyDgramObject* pyDgram, const char* baseName, Alg& alg) {
    const char* algName = alg.name();
    const uint32_t _v = alg.version();
    char keyName[TMPSTRINGSIZE];

    PyObject* software = Py_BuildValue("s", algName);
    PyObject* version  = Py_BuildValue("iii", (_v>>16)&0xff, (_v>>8)&0xff, (_v)&0xff);

    snprintf(keyName,TMPSTRINGSIZE,"software_%s_%s",baseName,"software");
    addObj(pyDgram, keyName, software);
    snprintf(keyName,TMPSTRINGSIZE,"software_%s_%s",baseName,"version");
    addObj(pyDgram, keyName, version);

    assert(Py_GETREF(software)==1);
}

static void setDetInfo(PyDgramObject* pyDgram, Names& names) {
    char keyName[TMPSTRINGSIZE];
    PyObject* detType = Py_BuildValue("s", names.detType());
    snprintf(keyName,TMPSTRINGSIZE,"software_%s_%s",names.detName(),"dettype");
    addObj(pyDgram, keyName, detType);

    PyObject* detId = Py_BuildValue("s", names.detId());
    snprintf(keyName,TMPSTRINGSIZE,"software_%s_%s",names.detName(),"detid");
    addObj(pyDgram, keyName, detId);

    assert(Py_GETREF(detType)==1);
}

void DictAssignAlg(PyDgramObject* pyDgram, std::vector<NameIndex>& namesVec)
{
    // This function gets called at configure: add attribute "software" and "version" to pyDgram and return
    char baseName[TMPSTRINGSIZE];

    for (unsigned i = 0; i < namesVec.size(); i++) {
        Names& names = namesVec[i].names();
        Alg& detAlg = names.alg();
        snprintf(baseName,TMPSTRINGSIZE,"%s_%s",names.detName(),names.alg().name());
        setAlg(pyDgram,baseName,detAlg);
        setDetInfo(pyDgram, names);

        for (unsigned j = 0; j < names.num(); j++) {
            Name& name = names.get(j);
            Alg& alg = name.alg();
            snprintf(baseName,TMPSTRINGSIZE,"%s_%s_%s",names.detName(),names.alg().name(),name.name());
            setAlg(pyDgram,baseName,alg);
        }
    }
}

void DictAssign(PyDgramObject* pyDgram, DescData& descdata)
{
    Names& names = descdata.nameindex().names(); // event names, chan0, chan1

    char keyName[TMPSTRINGSIZE];
    for (unsigned i = 0; i < names.num(); i++) {
        Name& name = names.get(i);
        const char* tempName = name.name();
        PyObject* newobj=0;

        if (name.rank() == 0) {
            switch (name.type()) {
            case Name::UINT8: {
	            const auto tempVal = descdata.get_value<uint8_t>(tempName);
                newobj = Py_BuildValue("i", tempVal);
                break;
            }
            case Name::UINT16: {
                const auto tempVal = descdata.get_value<uint16_t>(tempName);
                newobj = Py_BuildValue("i", tempVal);
                break;
            }
            case Name::UINT32: {
                const auto tempVal = descdata.get_value<uint32_t>(tempName);
                newobj = Py_BuildValue("i", tempVal);
                break;
            }
            case Name::UINT64: {
                const auto tempVal = descdata.get_value<uint64_t>(tempName);
                newobj = Py_BuildValue("i", tempVal);
                break;
            }
            case Name::INT8: {
                const auto tempVal = descdata.get_value<int8_t>(tempName);
                newobj = Py_BuildValue("i", tempVal);
                break;
            }
            case Name::INT16: {
                const auto tempVal = descdata.get_value<int16_t>(tempName);
                newobj = Py_BuildValue("i", tempVal);
                break;
            }
            case Name::INT32: {
                const auto tempVal = descdata.get_value<int32_t>(tempName);
                newobj = Py_BuildValue("i", tempVal);
                break;
            }
            case Name::INT64: {
                const auto tempVal = descdata.get_value<int64_t>(tempName);
                newobj = Py_BuildValue("i", tempVal);
                break;
            }
            case Name::FLOAT: {
                const auto tempVal = descdata.get_value<float>(tempName);
                newobj = Py_BuildValue("f", tempVal);
                break;
            }
            case Name::DOUBLE: {
                const auto tempVal = descdata.get_value<double>(tempName);
                newobj = Py_BuildValue("d", tempVal);
                break;
            }
            }
        } else {
            npy_intp dims[name.rank()];
            uint32_t* shape = descdata.shape(name);
            for (unsigned j = 0; j < name.rank(); j++) {
                dims[j] = shape[j];
            }
            switch (name.type()) {
            case Name::UINT8: {
                auto arr = descdata.get_array<uint8_t>(i);
                newobj = PyArray_SimpleNewFromData(name.rank(), dims,
                                                   NPY_UINT8, arr.data());
                break;
            }
            case Name::UINT16: {
                auto arr = descdata.get_array<uint16_t>(i);
                newobj = PyArray_SimpleNewFromData(name.rank(), dims,
                                                   NPY_UINT16, arr.data());
                break;
            }
            case Name::UINT32: {
                auto arr = descdata.get_array<uint32_t>(i);
                newobj = PyArray_SimpleNewFromData(name.rank(), dims,
                                                   NPY_UINT32, arr.data());
                break;
            }
            case Name::UINT64: {
                auto arr = descdata.get_array<uint64_t>(i);
                newobj = PyArray_SimpleNewFromData(name.rank(), dims,
                                                   NPY_UINT64, arr.data());
                break;
            }
            case Name::INT8: {
                auto arr = descdata.get_array<int8_t>(i);
                newobj = PyArray_SimpleNewFromData(name.rank(), dims,
                                                   NPY_INT8, arr.data());
                break;
            }
            case Name::INT16: {
                auto arr = descdata.get_array<int16_t>(i);
                newobj = PyArray_SimpleNewFromData(name.rank(), dims,
                                                   NPY_INT16, arr.data());
                break;
            }
            case Name::INT32: {
                auto arr = descdata.get_array<int32_t>(i);
                newobj = PyArray_SimpleNewFromData(name.rank(), dims,
                                                   NPY_INT32, arr.data());
                break;
            }
            case Name::INT64: {
                auto arr = descdata.get_array<int64_t>(i);
                newobj = PyArray_SimpleNewFromData(name.rank(), dims,
                                                   NPY_INT64, arr.data());
                break;
            }
            case Name::FLOAT: {
                auto arr = descdata.get_array<float>(i);
                newobj = PyArray_SimpleNewFromData(name.rank(), dims,
                                                   NPY_FLOAT, arr.data());
                break;
            }
            case Name::DOUBLE: {
                auto arr = descdata.get_array<double>(i);
                newobj = PyArray_SimpleNewFromData(name.rank(), dims,
                                                   NPY_DOUBLE, arr.data());
                break;
            }
            }
            //clear NPY_ARRAY_WRITEABLE flag
            PyArray_CLEARFLAGS((PyArrayObject*)newobj, NPY_ARRAY_WRITEABLE);
        }
        snprintf(keyName,TMPSTRINGSIZE,"%s_%s_%s",names.detName(),names.alg().name(),
                 name.name());
        addObj(pyDgram, keyName, newobj);
    }
}

class PyConvertIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    PyConvertIter(Xtc* xtc, PyDgramObject* pyDgram, std::vector<NameIndex>& namesVec) :
        XtcIterator(xtc), _pyDgram(pyDgram), _namesVec(namesVec)
    {
    }

    int process(Xtc* xtc)
    {
        switch (xtc->contains.id()) { //enum Type { Parent, ShapesData, Shapes, Data, Names, NumberOf };
        case (TypeId::Parent): {
            iterate(xtc); // look inside anything that is a Parent
            break;
        }
        case (TypeId::ShapesData): {
            ShapesData& shapesdata = *(ShapesData*)xtc;
            // lookup the index of the names we are supposed to use
            unsigned namesId = shapesdata.shapes().namesId();
            // protect against the fact that this datagram
            // may not have a _namesVec
            if (namesId<_namesVec.size()) {
                DescData descdata(shapesdata, _namesVec[namesId]);
                DictAssign(_pyDgram, descdata);
            }
            break;
        }
        default:
            break;
        }
        return Continue;
    }

private:
    PyDgramObject*          _pyDgram;
    std::vector<NameIndex>& _namesVec; // need one of these for each source
};

void AssignDict(PyDgramObject* self, PyObject* configDgram) {
    bool isConfig;
    isConfig = (configDgram == 0) ? true : false;
    
    if (isConfig) configDgram = (PyObject*)self; // we weren't passed a config, so we must be config
    
    NamesIter namesIter(&((PyDgramObject*)configDgram)->dgram()->xtc);
    namesIter.iterate();
    
    if (isConfig) DictAssignAlg((PyDgramObject*)configDgram, namesIter.namesVec());
    
    PyConvertIter iter(&self->dgram()->xtc, self, namesIter.namesVec());
    iter.iterate();
}

static void dgram_dealloc(PyDgramObject* self)
{
    // cpo: this should not need to be XDECREF for pyseq.  how are
    // we creating dgrams with a NULL value for pyseq?
    Py_XDECREF(self->pyseq);
    Py_XDECREF(self->dict);
#ifndef PSANA_USE_LEGION
    if (self->buf.buf == NULL) {
        free(self->dgram());
    } else {
        PyBuffer_Release(&(self->buf));
    }
#else
    // In-place destructor is necessary because the dgram is deallocated with free below.
    self->array.~LegionArray();
#endif
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* dgram_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    PyDgramObject* self;
    self = (PyDgramObject*)type->tp_alloc(type, 0);
    if (self != NULL) self->dict = PyDict_New();
    return (PyObject*)self;
}

#ifdef PSANA_USE_LEGION
class LegionDgramRead : public LegionTask<LegionDgramRead, int, ssize_t, off_t> {
public:
    LegionDgramRead() {}

    LegionDgramRead(int fd, LegionArray &array, ssize_t dgram_size, off_t offset)
      : LegionTask(fd, dgram_size, offset)
    {
        add_array(array);
    }

    void run(int fd, ssize_t dgram_size, off_t offset)
    {
        Dgram *dgram = (Dgram *)arrays[0].get_pointer();
        int readSuccess = pread(fd, dgram, dgram_size, offset);
        if (readSuccess <= 0) {
            abort(); // Elliott: need better error handling for asynchronous case
        }
    }

    static Legion::TaskID task_id;
};

Legion::TaskID LegionDgramRead::task_id = LegionDgramRead::register_task("dgram_read");
#endif

static int dgram_read(PyDgramObject* self, ssize_t dgram_size, int sequential)
{
    int readSuccess=0;
    if (sequential) {
        readSuccess = read_dgram(self); // for read sequentially
        if (readSuccess != 0) {
            PyErr_SetString(PyExc_StopIteration, "loading self->dgram() was unsuccessful");
            return -1;
        }
    } else {
        off_t fOffset = (off_t)self->offset;
#ifndef PSANA_USE_LEGION
        readSuccess = pread(self->file_descriptor, self->dgram(), dgram_size, fOffset); // for read with offset
        if (readSuccess <= 0) {
            char s[TMPSTRINGSIZE];
            snprintf(s, TMPSTRINGSIZE, "loading self->dgram() was unsuccessful -- %s", strerror(errno));
            PyErr_SetString(PyExc_StopIteration, s);
            return -1;
        }
#else
        LegionDgramRead task(self->file_descriptor, self->array, dgram_size, fOffset);
        task.launch();
#endif
    }
    return 0;
}

static int dgram_init(PyDgramObject* self, PyObject* args, PyObject* kwds)
{
    static char* kwlist[] = {(char*)"file_descriptor",
                             (char*)"config",
                             (char*)"offset",
                             (char*)"size",
                             (char*)"view",
                             NULL};

    int fd=-1;
    PyObject* configDgram=0;
    self->offset=0;
    ssize_t dgram_size=0;
    bool isView=0;
    PyObject* view=0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
                                     "|iOllO", kwlist,
                                     &fd,
                                     &configDgram,
                                     &self->offset,
                                     &dgram_size,
                                     &view)) {
        return -1;
    }
    
    if (fd > -1) {
        if (fcntl(fd, F_GETFD) == -1) {
            PyErr_SetString(PyExc_OSError, "invalid file descriptor");
            return -1;
        }
    }

    isView = (view!=0) ? true : false;

    if (self->offset == -1) {
        buffered_reader_free(self->reader);
        PyErr_SetNone(PyExc_StopIteration); // fixme: use shared_ptr
        return -1;
    }

    if (!isView) {
#ifndef PSANA_USE_LEGION
        self->dgram() = (Dgram*)malloc(BUFSIZE);
#else
        self->array = LegionArray(BUFSIZE);
        self->is_dgram_valid = false;
#endif
    } else {
        if (PyObject_GetBuffer(view, &(self->buf), PyBUF_SIMPLE) == -1) {
            PyErr_SetString(PyExc_MemoryError, "unable to create dgram with the given view");
            return -1;
        }
        self->dgram() = (Dgram*)(((char *)self->buf.buf) + self->offset);
    }

    // Avoid blocking to check the pointer in the Legion case
#ifndef PSANA_USE_LEGION
    if (self->dgram() == NULL) {
        PyErr_SetString(PyExc_MemoryError, "insufficient memory to create Dgram object");
        return -1;
    }
#endif

    // The pyseq fields are initialized in dgram_get_*
    self->is_pyseq_valid = false;

    // Read the data if this dgram is not a view
    if (!isView) {
        if (configDgram == 0) {
            if (fd > -1) {
                // this is server reading config.
                // allocates a buffer for reading offsets
                self->reader = buffered_reader_new(fd);
            }
        } else {
            PyDgramObject* _configDgram = (PyDgramObject*)configDgram;
            self->reader = _configDgram->reader;
        }

        if (fd==-1 && configDgram==0) {
            self->dgram()->xtc.extent = 0; // for empty dgram
        } else {
            if (fd==-1) {
                self->file_descriptor=((PyDgramObject*)configDgram)->file_descriptor;
            } else {
                self->file_descriptor=fd;
            }

            bool sequential = (fd==-1) != (configDgram==0);
            int err = dgram_read(self, dgram_size, sequential);
            if (err) return err;
        }
    }
    AssignDict(self, configDgram);

    return 0;
}

#if PY_MAJOR_VERSION < 3
static Py_ssize_t PyDgramObject_getsegcount(PyDgramObject *self, Py_ssize_t *lenp) {
    return 1; // only supports single segment
}

static Py_ssize_t PyDgramObject_getreadbuf(PyDgramObject *self, Py_ssize_t segment, void **ptrptr) {
    *ptrptr = (void*)self->dgram();
    if (self->dgram()->xtc.extent == 0) {
        return BUFSIZE;
    } else {
        return sizeof(*self->dgram()) + self->dgram()->xtc.sizeofPayload();
    }
}

static Py_ssize_t PyDgramObject_getwritebuf(PyDgramObject *self, Py_ssize_t segment, void **ptrptr) {
    return PyDgramObject_getreadbuf(self, segment, (void **)ptrptr);
}

static Py_ssize_t PyDgramObject_getcharbuf(PyDgramObject *self, Py_ssize_t segment, constchar **ptrptr) {
    return PyDgramObject_getreadbuf(self, segment, (void **) ptrptr);
}
#endif /* PY_MAJOR_VERSION < 3 */

static int PyDgramObject_getbuffer(PyObject *obj, Py_buffer *view, int flags)
{
    if (view == 0) {
        PyErr_SetString(PyExc_ValueError, "NULL view in getbuffer");
        return -1;
    }

    PyDgramObject* self = (PyDgramObject*)obj;
    view->obj = (PyObject*)self;
    view->buf = (void*)self->dgram();
    if (self->dgram()->xtc.extent == 0) {
        view->len = BUFSIZE; // share max size for empty dgram 
    } else {
        view->len = sizeof(*self->dgram()) + self->dgram()->xtc.sizeofPayload();
    }
    view->readonly = 1;
    view->itemsize = 1;
    view->format = (char *)"s";
    view->ndim = 1;
    view->shape = &view->len;
    view->strides = &view->itemsize;
    view->suboffsets = NULL;
    view->internal = NULL;
    
    Py_INCREF(self);  
    return 0;    
}

static PyObject* dgram_get_seq(PyDgramObject* self, void *closure) {
    if (!self->is_pyseq_valid) {
        //cpo: wasteful to do the Import here every time?
        PyObject* seqmod = PyImport_ImportModule("psana.seq");
        PyObject* pyseqtype = PyObject_GetAttrString(seqmod,"Seq");
        PyObject* capsule = PyCapsule_New((void*)&(self->dgram()->seq), NULL, NULL);
        PyObject* arglist = Py_BuildValue("(O)", capsule);
        Py_DECREF(capsule); // now owned by the arglist
        Py_DECREF(seqmod);
        Py_DECREF(pyseqtype);
        self->pyseq = PyObject_CallObject(pyseqtype, arglist);
        self->is_pyseq_valid = true;
    }

    Py_INCREF(self->pyseq);
    return self->pyseq;
}

static PyBufferProcs PyDgramObject_as_buffer = {
#if PY_MAJOR_VERSION < 3
    (readbufferproc)PyDgramObject_getreadbuf,   /*bf_getreadbuffer*/
    (writebufferproc)PyDgramObject_getwritebuf, /*bf_getwritebuffer*/
    (segcountproc)PyDgramObject_getsegcount,    /*bf_getsegcount*/
    (charbufferproc)PyDgramObject_getcharbuf,   /*bf_getcharbuffer*/
#endif
    // this definition is only compatible with Python 3.3 and above
    (getbufferproc)PyDgramObject_getbuffer,
    (releasebufferproc)0, // no special release required
};

static PyMemberDef dgram_members[] = {
    { (char*)"__dict__",
      T_OBJECT_EX, offsetof(PyDgramObject, dict),
      0,
      (char*)"attribute dictionary" },
    { (char*)"_file_descriptor",
      T_INT, offsetof(PyDgramObject, file_descriptor),
      0,
      (char*)"attribute file_descriptor" },
    { (char*)"_offset",
      T_INT, offsetof(PyDgramObject, offset),
      0,
      (char*)"attribute offset" },
    { NULL }
};

static PyGetSetDef dgram_getset[] = {
    { (char*)"seq",
      (getter)dgram_get_seq,
      NULL,
      (char*)"Dgram::Sequence",
      NULL },
    { NULL }
};

static PyObject* dgram_assign_dict(PyDgramObject* self) {
    AssignDict(self, 0); // Todo: may need to be fixed for other non-config dgrams
    Py_RETURN_NONE;
}

static PyMethodDef dgram_methods[] = {
    {"_assign_dict", (PyCFunction)dgram_assign_dict, METH_NOARGS,
     "Assign dictionary to the dgram"
    },
    {NULL}  /* Sentinel */
};

static PyTypeObject dgram_DgramType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "dgram.Dgram", /* tp_name */
    sizeof(PyDgramObject), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)dgram_dealloc, /* tp_dealloc */
    0, /* tp_print */
    0, /* tp_getattr */
    0, /* tp_setattr */
    0, /* tp_compare */
    0, /* tp_repr */
    0, /* tp_as_number */
    0, /* tp_as_sequence */
    0, /* tp_as_mapping */
    0, /* tp_hash */
    0, /* tp_call */
    0, /* tp_str */
    0, /* tp_getattro */
    0, /* tp_setattro */
    &PyDgramObject_as_buffer, /* tp_as_buffer */
    (Py_TPFLAGS_DEFAULT 
#if PY_MAJOR_VERSION < 3
    | Py_TPFLAGS_CHECKTYPES
    | Py_TPFLAGS_HAVE_NEWBUFFER
#endif
    | Py_TPFLAGS_BASETYPE),   /* tp_flags */
    0, /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    0, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    0, /* tp_iter */
    0, /* tp_iternext */
    dgram_methods, /* tp_methods */
    dgram_members, /* tp_members */
    dgram_getset, /* tp_getset */
    0, /* tp_base */
    0, /* tp_dict */
    0, /* tp_descr_get */
    0, /* tp_descr_set */
    offsetof(PyDgramObject, dict), /* tp_dictoffset */
    (initproc)dgram_init, /* tp_init */
    0, /* tp_alloc */
    dgram_new, /* tp_new */
    0, /* tp_free;  Low-level free-memory routine */
    0, /* tp_is_gc;  For PyObject_IS_GC */
    0, /* tp_bases*/
    0, /* tp_mro;  method resolution order */
    0, /* tp_cache*/
    0, /* tp_subclasses*/
    0, /* tp_weaklist*/
    (destructor)dgram_dealloc, /* tp_del*/
};

#if PY_MAJOR_VERSION > 2
static PyModuleDef dgrammodule =
{ PyModuleDef_HEAD_INIT, "dgram", NULL, -1, NULL, NULL, NULL, NULL, NULL };
#endif

#ifndef PyMODINIT_FUNC /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

#if PY_MAJOR_VERSION > 2
PyMODINIT_FUNC PyInit_dgram(void)
{
    PyObject* m;

    import_array();

    if (PyType_Ready(&dgram_DgramType) < 0) {
        return NULL;
    }

    m = PyModule_Create(&dgrammodule);
    if (m == NULL) {
        return NULL;
    }

    Py_INCREF(&dgram_DgramType);
    PyModule_AddObject(m, "Dgram", (PyObject*)&dgram_DgramType);
    return m;
}
#else
PyMODINIT_FUNC initdgram(void) {
    PyObject *m;
    
    import_array();

    if (PyType_Ready(&dgram_DgramType) < 0)
        return;

    m = Py_InitModule3("dgram", dgram_methods, "Dgram module.");

    if (m == NULL)
        return;

    Py_INCREF(&dgram_DgramType);
    PyModule_AddObject(m, "Dgram", (PyObject *)&dgram_DgramType);
}
#endif
