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
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <structmember.h>

#ifdef PSANA_USE_LEGION
#define LEGION_ENABLE_C_BINDINGS
#include <legion.h>
#include <legion/legion_c_util.h>
using namespace Legion;
#endif

using namespace XtcData;
#define BUFSIZE 0x4000000
#define TMPSTRINGSIZE 256

#ifdef PSANA_USE_LEGION
enum FieldIDs {
  FID_X = 101,
};
#endif

// to avoid compiler warnings for debug variables
#define _unused(x) ((void)(x))

typedef struct {
    PyObject_HEAD
    PyObject* dict;
    Dgram* dgram;
#ifdef PSANA_USE_LEGION
    LogicalRegionT<1> region;
    PhysicalRegion physical;
#endif
    int file_descriptor;
    int verbose;
    int debug;
    int offset;
} PyDgramObject;

static void write_object_info(PyDgramObject* self, PyObject* obj, const char* comment)
{
    if (self->verbose > 0) {
        printf("VERBOSE=%d; %s\n", self->verbose, comment);
        fflush(stdout);
        printf("VERBOSE=%d;  self->debug=%d\n", self->verbose, self->debug);
        printf("VERBOSE=%d;  self=%p\n", self->verbose, self);
        printf("VERBOSE=%d;  Py_REFCNT(self)=%d\n", self->verbose, (int)Py_REFCNT(self));
        fflush(stdout);
    }
}

static void setAlg(PyDgramObject* pyDgram, const char* baseName, Alg& alg) {
    const char* algName = alg.name();
    const uint32_t _v = alg.version();
    char keyName[TMPSTRINGSIZE];

    PyObject* newobjS = Py_BuildValue("s", algName);
    PyObject* newobjV= Py_BuildValue("iii", (_v>>16)&0xff, (_v>>8)&0xff, (_v)&0xff);

    snprintf(keyName,TMPSTRINGSIZE,"software_%s_%s",baseName,"software");
    PyObject* keyS = PyUnicode_FromString(keyName);
    if (PyDict_Contains(pyDgram->dict, keyS)) {
        printf("Dgram: Ignoring duplicate key %s\n", keyName);
    } else {
        PyDict_SetItem(pyDgram->dict, keyS, newobjS);
        Py_DECREF(newobjS);
        snprintf(keyName,TMPSTRINGSIZE,"software_%s_%s",baseName,"version");
        PyObject* keyV = PyUnicode_FromString(keyName);
        PyDict_SetItem(pyDgram->dict, keyV, newobjV);
        Py_DECREF(newobjV);
    }
}

static void setDetInfo(PyDgramObject* pyDgram, Names& names) {
    char keyName[TMPSTRINGSIZE];
    PyObject* newobjDetType = Py_BuildValue("s", names.detType());
    snprintf(keyName,TMPSTRINGSIZE,"software_%s_%s",names.detName(),"dettype");
    PyObject* keyDetType = PyUnicode_FromString(keyName);
    if (PyDict_Contains(pyDgram->dict, keyDetType)) {
        // this will happen since the same detname/dettype pair
        // show up once for every Names object.
    } else {
        PyDict_SetItem(pyDgram->dict, keyDetType, newobjDetType);
        Py_DECREF(newobjDetType);
    }

    newobjDetType = Py_BuildValue("s", names.detId());
    snprintf(keyName,TMPSTRINGSIZE,"software_%s_%s",names.detName(),"detid");
    keyDetType = PyUnicode_FromString(keyName);
    if (PyDict_Contains(pyDgram->dict, keyDetType)) {
        // this will happen since the same detname/dettype pair
        // show up once for every Names object.
    } else {
        PyDict_SetItem(pyDgram->dict, keyDetType, newobjDetType);
        Py_DECREF(newobjDetType);
    }
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
        PyObject* newobj;

        if (name.rank() == 0) {
            switch (name.type()) {
            case Name::UINT8: {
                const int tempVal = descdata.get_value<uint8_t>(tempName);
                newobj = Py_BuildValue("i", tempVal);
                break;
            }
            case Name::UINT16: {
                const int tempVal = descdata.get_value<uint16_t>(tempName);
                newobj = Py_BuildValue("i", tempVal);
                break;
            }
            case Name::INT32: {
                const int tempVal = descdata.get_value<int32_t>(tempName);
                newobj = Py_BuildValue("i", tempVal);
                break;
            }
            case Name::FLOAT: {
                const float tempVal = descdata.get_value<float>(tempName);
                newobj = Py_BuildValue("f", tempVal);
                break;
            }
            case Name::DOUBLE: {
                const double tempVal = descdata.get_value<double>(tempName);
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
                newobj = PyArray_SimpleNewFromData(name.rank(), dims,
                                                   NPY_UINT8, descdata.address(i));
                break;
            }
            case Name::UINT16: {
                newobj = PyArray_SimpleNewFromData(name.rank(), dims,
                                                   NPY_UINT16, descdata.address(i));
                break;
            }
            case Name::INT32: {
                newobj = PyArray_SimpleNewFromData(name.rank(), dims,
                                                   NPY_INT32, descdata.address(i));
                break;
            }
            case Name::FLOAT: {
                newobj = PyArray_SimpleNewFromData(name.rank(), dims,
                                                   NPY_FLOAT, descdata.address(i));
                break;
            }
            case Name::DOUBLE: {
                newobj = PyArray_SimpleNewFromData(name.rank(), dims,
                                                   NPY_DOUBLE, descdata.address(i));
                break;
            }
            }
            if ( (pyDgram->debug & 0x01) != 0 ) {
                // place holder for old-style pointer management -- this should be remove at some point
                printf("Warning: using old-style pointer management in DictAssign() (i.e. debug=1)\n");
            } else {
                // New default behaviour
                if (PyArray_SetBaseObject((PyArrayObject*)newobj, (PyObject*)pyDgram) < 0) {
                    char s[TMPSTRINGSIZE];
                    snprintf(s, TMPSTRINGSIZE, "Failed to set BaseObject for numpy array (%s)\n", strerror(errno));
                    PyErr_SetString(PyExc_StopIteration, s);
                    return;
                }
                Py_INCREF(pyDgram);
            }
            //clear NPY_ARRAY_WRITEABLE flag
            PyArray_CLEARFLAGS((PyArrayObject*)newobj, NPY_ARRAY_WRITEABLE);
        }
        snprintf(keyName,TMPSTRINGSIZE,"%s_%s_%s",names.detName(),names.alg().name(),
                 name.name());
        PyObject* key = PyUnicode_FromString(keyName);
        if (PyDict_Contains(pyDgram->dict, key)) {
            printf("Dgram: Ignoring duplicate key %s\n", tempName);
        } else {
            PyDict_SetItem(pyDgram->dict, key, newobj);
            // when the new objects are created they get a reference
            // count of 1.  PyDict_SetItem increases this to 2.  we
            // decrease it back to 1 here, which effectively gives
            // ownership of the new objects to the dictionary.  when
            // the dictionary is deleted, the objects will be deleted.
            Py_DECREF(newobj);
        }
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
            DescData descdata(shapesdata, _namesVec[namesId]);
            DictAssign(_pyDgram, descdata);
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

static void dgram_dealloc(PyDgramObject* self)
{
    write_object_info(self, NULL, "Top of dgram_dealloc()");
    Py_XDECREF(self->dict);
#ifndef PSANA_USE_LEGION
    free(self->dgram);
#else
    {
      Runtime *runtime = Runtime::get_runtime();
      Context ctx = Runtime::get_context();

      // FIXME: Causes runtime type error
      // self->physical = PhysicalRegion();
      runtime->destroy_logical_region(ctx, self->region);
    }
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

static int dgram_init(PyDgramObject* self, PyObject* args, PyObject* kwds)
{
    static char* kwlist[] = {(char*)"file_descriptor",
                             (char*)"config",
                             (char*)"verbose",
                             (char*)"debug",
                             (char*)"offset",
                             NULL};
    int fd=0;
    bool isConfig;
    PyObject* configDgram=0;
    self->verbose=0;
    self->debug=0;
    self->offset=0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
                                     "|iO$iii", kwlist,
                                     &fd,
                                     &configDgram,
                                     &(self->verbose),
                                     &(self->debug),
                                     &(self->offset))) {
        return -1;
    }
    isConfig = (configDgram==0) ? true : false;

    if (fd==0 && configDgram==0) {
        PyErr_SetString(PyExc_StopIteration, "Must specify either file_descriptor or config");
        return -1;
    }

    if (fd==0) {
        fd=((PyDgramObject*)configDgram)->file_descriptor;
    } else {
        self->file_descriptor=fd;
    }

#ifndef PSANA_USE_LEGION
    self->dgram = (Dgram*)malloc(BUFSIZE);
#else
    {
      Runtime *runtime = Runtime::get_runtime();
      Context ctx = Runtime::get_context();

      IndexSpaceT<1> ispace = runtime->create_index_space(ctx, Rect<1>(0, BUFSIZE-1));
      FieldSpace fspace = runtime->create_field_space(ctx);
      FieldAllocator falloc = runtime->create_field_allocator(ctx, fspace);
      falloc.allocate_field(1, FID_X);
      self->region = runtime->create_logical_region(ctx, ispace, fspace);

      InlineLauncher launcher(RegionRequirement(self->region, READ_WRITE, EXCLUSIVE, self->region));
      launcher.add_field(FID_X);
      self->physical = runtime->map_region(ctx, launcher);
      self->physical.wait_until_valid();
      UnsafeFieldAccessor<char,1,coord_t,Realm::AffineAccessor<char,1,coord_t> > acc(self->physical, FID_X);
      self->dgram = (Dgram*)acc.ptr(0);
    }
#endif
    if (self->dgram == NULL) {
        PyErr_SetString(PyExc_MemoryError, "insufficient memory to create Dgram object");
        return -1;
    }
    
    off_t fOffset = (off_t)self->offset;
    int readSuccess=0;
    if (fOffset == 0) { 
        readSuccess = ::read(fd, self->dgram, sizeof(*self->dgram));
    } else {
        readSuccess = ::pread(fd, self->dgram, sizeof(*self->dgram), fOffset);
    }
    if (readSuccess <= 0) {
        char s[TMPSTRINGSIZE];
        snprintf(s, TMPSTRINGSIZE, "loading self->dgram was unsuccessful -- %s", strerror(errno));
        PyErr_SetString(PyExc_StopIteration, s);
        return -1;
    }
    
    size_t payloadSize = self->dgram->xtc.sizeofPayload();
    readSuccess = 0;
    if (fOffset == 0) {
        readSuccess = ::read(fd, self->dgram->xtc.payload(), payloadSize);
    } else { 
        fOffset += (off_t)sizeof(*self->dgram);
        readSuccess = ::pread(fd, self->dgram->xtc.payload(), payloadSize, fOffset);
    }
    if (readSuccess <= 0){
        char s[TMPSTRINGSIZE];
        snprintf(s, TMPSTRINGSIZE, "loading self->dgram->xtc.payload() was unsuccessful -- %s", strerror(errno));
        PyErr_SetString(PyExc_StopIteration, s);
        return -1;
    }

    if (isConfig) configDgram = (PyObject*)self; // we weren't passed a config, so we must be config

    NamesIter namesIter(&((PyDgramObject*)configDgram)->dgram->xtc);
    namesIter.iterate();

    if (isConfig) DictAssignAlg((PyDgramObject*)configDgram, namesIter.namesVec());

    PyConvertIter iter(&self->dgram->xtc, self, namesIter.namesVec());
    iter.iterate();

    return 0;
}

static PyMemberDef dgram_members[] = {
    { (char*)"__dict__",
      T_OBJECT_EX, offsetof(PyDgramObject, dict),
      0,
      (char*)"attribute dictionary" },
    { (char*)"file_descriptor",
      T_INT, offsetof(PyDgramObject, file_descriptor),
      0,
      (char*)"attribute file_descriptor" },
    { (char*)"verbose",
      T_INT, offsetof(PyDgramObject, verbose),
      0,
      (char*)"attribute verbose" },
    { (char*)"debug",
      T_INT, offsetof(PyDgramObject, debug),
      0,
      (char*)"attribute debug" },
    { (char*)"offset",
      T_INT, offsetof(PyDgramObject, offset),
      0,
      (char*)"attribute offset" },
    { NULL }
};


PyObject* tp_getattro(PyObject* obj, PyObject* key)
{
    PyObject* res = PyDict_GetItem(((PyDgramObject*)obj)->dict, key);
    if (res != NULL) {
        if ( (((PyDgramObject*)obj)->debug & 0x01) != 0 ) {
            // old-style pointer management -- this should be remove at some point
            printf("Warning: using old-style pointer management in tp_getattro() (i.e. debug=1)\n");
            if (strcmp("numpy.ndarray", res->ob_type->tp_name) == 0) {
                PyArrayObject* arr = (PyArrayObject*)res;
                PyObject* arr_copy = PyArray_SimpleNewFromData(PyArray_NDIM(arr), PyArray_DIMS(arr),
                                                               PyArray_DESCR(arr)->type_num, PyArray_DATA(arr));
                if (PyArray_SetBaseObject((PyArrayObject*)arr_copy, (PyObject*)obj) < 0) {
                    printf("Failed to set BaseObject for numpy array.\n");
                    return 0;
                }
                // this reference count will get decremented when the returned
                // array is deleted (since the array has us as the "base" object).
                Py_INCREF(obj);
                //return arr_copy;
                res=arr_copy;
            } else {
                // this reference count will get decremented when the returned
                // variable is deleted, so must increment here.
                Py_INCREF(res);
            }
        } else {
            // New default behaviour
            Py_INCREF(res);
            PyDict_DelItem(((PyDgramObject*)obj)->dict, key);
        }
    } else {
        res = PyObject_GenericGetAttr(obj, key);
    }

    return res;
}

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
    tp_getattro, /* tp_getattro */
    0, /* tp_setattro */
    0, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    0, /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    0, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    0, /* tp_iter */
    0, /* tp_iternext */
    0, /* tp_methods */
    dgram_members, /* tp_members */
    0, /* tp_getset */
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

static PyModuleDef dgrammodule =
{ PyModuleDef_HEAD_INIT, "dgram", NULL, -1, NULL, NULL, NULL, NULL, NULL };

#ifndef PyMODINIT_FUNC /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
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
