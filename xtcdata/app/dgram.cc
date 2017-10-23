#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/TypeId.hh"
#include "xtcdata/xtc/XtcIterator.hh"

#include <Python.h>
#include <stdio.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <errno.h>
#include <fcntl.h>
#include <numpy/arrayobject.h>
#include <structmember.h>

using namespace XtcData;
#define BUFSIZE 0x4000000

// to avoid compiler warnings for debug variables
#define _unused(x) ((void)(x))

typedef struct {
    PyObject_HEAD
    PyObject* dict;
    Dgram* dgram;
    int verbose;
    int debug;
} PyDgramObject;

static void write_object_info(PyDgramObject* self, PyObject* obj, const char* comment)
{
    if (self->verbose > 0) {
        printf("VERBOSE=%d; %s\n", self->verbose, comment);
        fflush(stdout);
        printf("VERBOSE=%d;  self=%p\n", self->verbose, self);
        printf("VERBOSE=%d;  Py_REFCNT(self)=%d\n", self->verbose, (int)Py_REFCNT(self));
        printf("VERBOSE=%d;  self->dict=%p\n", self->verbose, self->dict);
        if (self->dict != NULL) {
            printf("VERBOSE=%d;  Py_REFCNT(self->dict)=%d\n", self->verbose, (int)Py_REFCNT(self->dict));
        }
        fflush(stdout);
        if (obj != NULL) {
            printf("VERBOSE=%d;  Py_REFCNT(obj)=%d\n", self->verbose, (int)Py_REFCNT(obj));
            if (strcmp("numpy.ndarray", obj->ob_type->tp_name) == 0) {
                PyArrayObject_fields* p=(PyArrayObject_fields*)obj;
                printf("VERBOSE=%d;  obj->base=%p\n", self->verbose, p->base);
                fflush(stdout);
                if (p->base != NULL) {
                    printf("VERBOSE=%d;  Py_REFCNT(obj->base)=%d\n", self->verbose, (int)Py_REFCNT(p->base));
                    fflush(stdout);
                }
            }
        }
        printf("VERBOSE=%d; %s\n\n", self->verbose, "done");
        fflush(stdout);
    }
}

void DictAssign(PyDgramObject* pyDgram, DescData& descdata)
{
    Names& names = descdata.nameindex().names();
    for (unsigned i = 0; i < names.num(); i++) {
        Name& name = names.get(i);
        const char* tempName = name.name();
        PyObject* key = PyUnicode_FromString(tempName);
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
            if ( (pyDgram->debug & 0x03) != 0 ) {
                if (PyArray_SetBaseObject((PyArrayObject*)newobj, (PyObject*)pyDgram) < 0) {
                    char s[120];
                    sprintf(s, "Failed to set BaseObject for numpy array (%s)\n", strerror(errno));
                    PyErr_SetString(PyExc_StopIteration, s);
                    return;
                }
                Py_INCREF(pyDgram);
            }
        }
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
        char s[120];
        sprintf(s, "Bottom of DictAssign, obj is %s", tempName);
        write_object_info(pyDgram, newobj, s);
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
        switch (xtc->contains.id()) {
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
    free(self->dgram);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* dgram_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    PyDgramObject* self;
    self = (PyDgramObject*)type->tp_alloc(type, 0);
    if (self != NULL) self->dict = PyDict_New();
    return (PyObject*)self;
}

class myNamesIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    myNamesIter(Xtc* xtc) : XtcIterator(xtc) {}

    int process(Xtc* xtc)
    {
        // printf("found typeid %s\n",XtcData::TypeId::name(xtc->contains.id()));
        switch (xtc->contains.id()) {
        case (TypeId::Parent): {
            iterate(xtc); // look inside anything that is a Parent
            break;
        }
        case (TypeId::Names): {
            _namesVec.push_back(NameIndex(*(Names*)xtc));
            break;
        }
        default:
            break;
        }
        return Continue;
    }
    std::vector<NameIndex>& namesVec() {return _namesVec;}
private:
    std::vector<NameIndex> _namesVec;
};

static int dgram_init(PyDgramObject* self, PyObject* args, PyObject* kwds)
{
    static char* kwlist[] = {(char*)"file_descriptor",
                             (char*)"config",
                             (char*)"verbose",
                             (char*)"debug",
                             NULL};
    int fd;
    PyObject* configDgram=0;
    self->verbose=0;
    self->debug=0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
                                     "i|O$ii", kwlist,
                                     &fd,
                                     &configDgram,
                                     &(self->verbose),
                                     &(self->debug))) {
        return -1;
    }

    self->dgram = (Dgram*)malloc(BUFSIZE);
    if (self->dgram == NULL) {
        PyErr_SetString(PyExc_MemoryError, "insufficient memory to create Dgram object");
        return -1;
    }

    if (::read(fd, self->dgram, sizeof(*self->dgram)) <= 0) {
        char s[120];
        sprintf(s, "loading self->dgram was unsuccessful -- %s", strerror(errno));
        PyErr_SetString(PyExc_StopIteration, s);
        return -1;
    }

    size_t payloadSize = self->dgram->xtc.sizeofPayload();
    if (::read(fd, self->dgram->xtc.payload(), payloadSize) <= 0) {
        char s[120];
        sprintf(s, "loading self->dgram->xtc.payload() was unsuccessful -- %s", strerror(errno));
        PyErr_SetString(PyExc_StopIteration, s);
        return -1;
    }

    if (configDgram==0) configDgram = (PyObject*)self; // we weren't passed a config, so we must be config
    myNamesIter namesIter(&((PyDgramObject*)configDgram)->dgram->xtc);
    namesIter.iterate();
    
    PyConvertIter iter(&self->dgram->xtc, self, namesIter.namesVec());
    iter.iterate();

    write_object_info(self, NULL, "Bottom of dgram_init()");
    return 0;
}

static PyMemberDef dgram_members[] = {
    { (char*)"__dict__",
      T_OBJECT_EX, offsetof(PyDgramObject, dict),
      0,
      (char*)"attribute dictionary" },
    { (char*)"verbose",
      T_INT, offsetof(PyDgramObject, verbose),
      0,
      (char*)"attribute verbose" },
    { (char*)"debug",
      T_INT, offsetof(PyDgramObject, debug),
      0,
      (char*)"attribute debug" },
    { NULL }
};

PyObject* tp_getattro(PyObject* o, PyObject* key)
{
    PyObject* res = PyDict_GetItem(((PyDgramObject*)o)->dict, key);
    char s[120];
    Py_ssize_t size;
    char *key_string = PyUnicode_AsUTF8AndSize(key, &size);
    sprintf(s, "Top of tp_getattro(), obj is %s", key_string);
    write_object_info((PyDgramObject*)o, res, s);

    if (res != NULL) {
        if ( (((PyDgramObject*)o)->debug & 0x01) == 1 ) {
            Py_INCREF(res);
            PyDict_DelItem(((PyDgramObject*)o)->dict, key);
            if (PyDict_Size(((PyDgramObject*)o)->dict) == 0) {
                Py_CLEAR(((PyDgramObject*)o)->dict);
            }
        } else if ( (((PyDgramObject*)o)->debug & 0x02) == 1 ) {
            Py_INCREF(res);
        } else {
            if (strcmp("numpy.ndarray", res->ob_type->tp_name) == 0) {
                PyArrayObject* arr = (PyArrayObject*)res;
                PyObject* arr_copy = PyArray_SimpleNewFromData(PyArray_NDIM(arr), PyArray_DIMS(arr),
                                                               PyArray_DESCR(arr)->type_num, PyArray_DATA(arr));
                if (PyArray_SetBaseObject((PyArrayObject*)arr_copy, (PyObject*)o) < 0) {
                    printf("Failed to set BaseObject for numpy array.\n");
                    return 0;
                }
                // this reference count will get decremented when the returned
                // array is deleted (since the array has us as the "base" object).
                Py_INCREF(o);
                //return arr_copy;
                res=arr_copy;
            } else {
                // this reference count will get decremented when the returned
                // variable is deleted, so must increment here.
                Py_INCREF(res);
            }
        }

    } else {
        res = PyObject_GenericGetAttr(o, key);
    }


    //char s[120];
    //Py_ssize_t size;
    //char *key_string = PyUnicode_AsUTF8AndSize(key, &size);
    sprintf(s, "Bottom of tp_getattro(), obj is %s", key_string);
    write_object_info((PyDgramObject*)o, res, s);
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
