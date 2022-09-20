#include <Python.h>
#include <stdio.h>
#include <stddef.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <structmember.h>

typedef struct {
    PyObject_HEAD
    PyObject* dict;
} PyContainerObject;

static void container_dealloc(PyContainerObject* self)
{
    Py_DECREF(self->dict);
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
    return 0;
}

// trying to following this example, but with less error checking
// https://pythonextensionpatterns.readthedocs.io/en/latest/pickle.html
static PyObject *
container__getstate__(PyContainerObject *self, PyObject *Py_UNUSED(ignored)) {
    //return self->dict;
    PyObject *ret = Py_BuildValue("{sO}", "container_dict", self->dict);
    return ret;
}

static PyObject *
container__setstate__(PyContainerObject *self, PyObject *state) {
    if (!PyDict_CheckExact(state)) {
        PyErr_SetString(PyExc_ValueError, "Pickled psana container object is not a dict.");
        return NULL;
    }
    Py_DECREF(self->dict);
    self->dict = PyDict_GetItemString(state, "container_dict"); /* Borrowed reference. */
    if (self->dict == NULL) {
        /* PyDict_GetItemString does not set any error state so we have to. */
        PyErr_SetString(PyExc_KeyError, "No \"container_dict\" key in pickled dict.");
        return NULL;
    }
    /* Increment the borrowed reference for our instance of it. */
    Py_INCREF(self->dict);
    Py_RETURN_NONE;
}

static PyMemberDef container_members[] = {
    { (char*)"__dict__",
      T_OBJECT_EX, offsetof(PyContainerObject, dict),
      0,
      (char*)"attribute dictionary" },
    { NULL }
};

static PyMethodDef container_methods[] = {
    {"__getstate__", (PyCFunction)container__getstate__, METH_NOARGS, "pickle the container object"},
    {"__setstate__", (PyCFunction)container__setstate__, METH_O, "unpickle the container object"},
    {NULL}  /* Sentinel */
};

static PyTypeObject container_ContainerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "psana.container.Container", /* tp_name */
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
    0, /* tp_getattro */
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
    container_methods, /* tp_methods */
    container_members, /* tp_members */
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
