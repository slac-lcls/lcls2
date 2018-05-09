#include <Python.h>
#include <stdio.h>
#include <stddef.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

typedef struct {
    PyObject_HEAD
    PyObject* dict;
    PyObject* dgram;
} PyContainerObject;

static void container_dealloc(PyContainerObject* self)
{
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* container_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    PyContainerObject* self;
    self = (PyContainerObject*)type->tp_alloc(type, 0);
    if (self != NULL) self->dict = PyDict_New();
    return (PyObject*)self;
}

static int container_init(PyContainerObject* self, PyObject* args, PyObject* kwds)
{
    PyArg_ParseTuple(args, "O", &(self->dgram));
    return 0;
}

static PyObject* tp_getattro(PyObject* self, PyObject* key)
{
    PyObject* res = PyDict_GetItem(((PyContainerObject*)self)->dict, key);
    if (res != NULL) {
        if (strcmp("numpy.ndarray", res->ob_type->tp_name) == 0) {
            PyArrayObject* arr = (PyArrayObject*)res;
            PyObject* arr_copy = PyArray_SimpleNewFromData(PyArray_NDIM(arr), PyArray_DIMS(arr),
                                                           PyArray_DESCR(arr)->type_num, PyArray_DATA(arr));
            if (PyArray_SetBaseObject((PyArrayObject*)arr_copy, ((PyContainerObject*)self)->dgram) < 0) {
                printf("Failed to set BaseObject for numpy array.\n");
                return 0;
            }
            // this reference count will get decremented when the returned
            // array is deleted (since the array has us as the "base" object).
            Py_INCREF(((PyContainerObject*)self)->dgram);
            res=arr_copy;
        } else {
            // this reference count will get decremented when the returned
            // variable is deleted, so must increment here.
            Py_INCREF(res);
        }
    } else {
        res = PyObject_GenericGetAttr(self, key);
    }

    return res;
}

static PyTypeObject container_ContainerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "container.Container", /* tp_name */
    sizeof(PyContainerObject), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)container_dealloc, /* tp_dealloc */
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
    (Py_TPFLAGS_DEFAULT 
#if PY_MAJOR_VERSION < 3
    | Py_TPFLAGS_CHECKTYPES
#endif
    | Py_TPFLAGS_BASETYPE),   /* tp_flags */
    0, /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    0, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    0, /* tp_iter */
    0, /* tp_iternext */
    0, /* tp_methods */
    0, /* tp_members */
    0, /* tp_getset */
    0, /* tp_base */
    0, /* tp_dict */
    0, /* tp_descr_get */
    0, /* tp_descr_set */
    offsetof(PyContainerObject, dict), /* tp_dictoffset */
    (initproc)container_init, /* tp_init */
    0, /* tp_alloc */
    container_new, /* tp_new */
    0, /* tp_free;  Low-level free-memory routine */
    0, /* tp_is_gc;  For PyObject_IS_GC */
    0, /* tp_bases*/
    0, /* tp_mro;  method resolution order */
    0, /* tp_cache*/
    0, /* tp_subclasses*/
    0, /* tp_weaklist*/
    (destructor)container_dealloc, /* tp_del*/
};

#if PY_MAJOR_VERSION > 2
static PyModuleDef containermodule =
{ PyModuleDef_HEAD_INIT, "container", NULL, -1, NULL, NULL, NULL, NULL, NULL };
#endif

#ifndef PyMODINIT_FUNC /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

#if PY_MAJOR_VERSION > 2
PyMODINIT_FUNC PyInit_container(void)
{
    PyObject* m;

    import_array();

    if (PyType_Ready(&container_ContainerType) < 0) {
        return NULL;
    }

    m = PyModule_Create(&containermodule);
    if (m == NULL) {
        return NULL;
    }

    Py_INCREF(&container_ContainerType);
    PyModule_AddObject(m, "Container", (PyObject*)&container_ContainerType);
    return m;
}
#else
PyMODINIT_FUNC initcontainer(void) {
    PyObject *m;
    
    if (PyType_Ready(&container_ContainerType) < 0)
        return;

    m = Py_InitModule3("container", container_methods, "Container module.");

    if (m == NULL)
        return;

    Py_INCREF(&container_ContainerType);
    PyModule_AddObject(m, "Container", (PyObject *)&container_ContainerType);
}
#endif
