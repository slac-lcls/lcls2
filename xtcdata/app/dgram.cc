#include "xtcdata/xtc/Descriptor.hh"
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

typedef struct {
    PyObject_HEAD PyObject* dict;
    Dgram* dgram;
} PyDgramObject;

void DictAssign(PyDgramObject* dgram, Descriptor& desc, DescData& d)
{
    for (unsigned i = 0; i < desc.num_fields(); i++) {
        Field& f = desc.get(i);
        const char* tempName = f.name;
        PyObject* key = PyUnicode_FromString(tempName);
        PyObject* newobj;
        if (f.rank == 0) {
            switch (f.type) {
            case UINT8: {
                const int tempVal = d.get_value<uint8_t>(tempName);
                newobj = Py_BuildValue("i", tempVal);
                break;
            }
            case UINT16: {
                const int tempVal = d.get_value<uint16_t>(tempName);
                newobj = Py_BuildValue("i", tempVal);
                break;
            }
            case INT32: {
                const int tempVal = d.get_value<int32_t>(tempName);
                newobj = Py_BuildValue("i", tempVal);
                break;
            }
            case FLOAT: {
                const float tempVal = d.get_value<float>(tempName);
                newobj = Py_BuildValue("f", tempVal);
                break;
            }
            case DOUBLE: {
                const int tempVal = d.get_value<double>(tempName);
                newobj = Py_BuildValue("d", tempVal);
                break;
            }
            }
        } else {
            npy_intp dims[f.rank + 1];
            for (unsigned i = 0; i < f.rank + 1; i++) {
                dims[i] = f.shape[i];
            }
            switch (f.type) {
            case UINT8: {
                newobj = PyArray_SimpleNewFromData(f.rank, dims, NPY_UINT8, d.data() + f.offset);
                break;
            }
            case UINT16: {
                newobj = PyArray_SimpleNewFromData(f.rank, dims, NPY_UINT16, d.data() + f.offset);
                break;
            }
            case INT32: {
                newobj = PyArray_SimpleNewFromData(f.rank, dims, NPY_INT32, d.data() + f.offset);
                break;
            }
            case FLOAT: {
                newobj = PyArray_SimpleNewFromData(f.rank, dims, NPY_FLOAT, d.data() + f.offset);
                break;
            }
            case DOUBLE: {
                newobj = PyArray_SimpleNewFromData(f.rank, dims, NPY_DOUBLE, d.data() + f.offset);
                break;
            }
            }
        }
        if (PyDict_Contains(dgram->dict, key)) {
            printf("Dgram: Ignoring duplicate key %s\n", tempName);
        } else {
            PyDict_SetItem(dgram->dict, key, newobj);
            // when the new objects are created they get a reference
            // count of 1.  PyDict_SetItem increases this to 2.  we
            // decrease it back to 1 here, which effectively gives
            // ownership of the new objects to the dictionary.  when
            // the dictionary is deleted, the objects will be deleted.
            Py_DECREF(newobj);
        }
    }
}

class myLevelIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    myLevelIter(Xtc* xtc, PyDgramObject* dgram) : XtcIterator(xtc), _dgram(dgram)
    {
    }

    int process(Xtc* xtc)
    {
        switch (xtc->contains.id()) {
        case (TypeId::Parent): {
            iterate(xtc); // look inside anything that is a Parent
            break;
        }
        case (TypeId::DescData): {
            DescData& d = *(DescData*)xtc->payload();
            Descriptor& desc = d.desc();
            DictAssign(_dgram, desc, d);
            break;
        }
        default:
            break;
        }
        return Continue;
    }

private:
    PyDgramObject* _dgram;
};

static void dgram_dealloc(PyDgramObject* self)
{
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

static int dgram_init(PyDgramObject* self, PyObject* args, PyObject* kwds)
{
    self->dgram = (Dgram*)malloc(BUFSIZE);

    int fd = open("data.xtc", O_RDONLY | O_LARGEFILE);

    if (::read(fd, self->dgram, sizeof(*self->dgram)) <= 0) {
        printf("read was unsuccessful: %s\n", strerror(errno));
    }

    size_t payloadSize = self->dgram->xtc.sizeofPayload();
    ::read(fd, self->dgram->xtc.payload(), payloadSize);

    myLevelIter iter(&self->dgram->xtc, self);
    iter.iterate();

    return 0;
}

static PyMemberDef dgram_members[] = { { (char*)"__dict__", T_OBJECT_EX, offsetof(PyDgramObject, dict),
                                         0, (char*)"attribute dictionary" },
                                       { NULL } };

PyObject* tp_getattro(PyObject* o, PyObject* key)
{
    PyObject* res = PyDict_GetItem(((PyDgramObject*)o)->dict, key);
    if (res != NULL) {
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
            return arr_copy;
        }
    } else {
        res = PyObject_GenericGetAttr(o, key);
    }

    return res;
}

static PyTypeObject dgram_DgramType = {
    PyVarObject_HEAD_INIT(NULL, 0) "dgram.Dgram", /* tp_name */
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
