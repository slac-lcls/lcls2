#include "structmember.h"
#include <Python.h>
#include <fcntl.h>
#include <numpy/arrayobject.h>

typedef struct {
    PyObject_HEAD int* data;
} Noddy;

static void Noddy_dealloc(Noddy* self)
{
    printf("Hit the Dealloc function\n\n");
    // Py_XDECREF(self->array);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* Noddy_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    Noddy* self;

    self = (Noddy*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->data = new int[10];
    }

    return (PyObject*)self;
}

static int Noddy_init(Noddy* self, PyObject* args, PyObject* kwds)
{

    for (int i = 0; i < 10; i++) {
        self->data[i] = i;
    }


    // int32_t array[] = {1,2,3,4,5,6};
    // npy_intp dims[] = {6};
    // self->array = PyArray_SimpleNewFromData(1,dims,NPY_INT32,array);
    // Py_INCREF((PyObject*)self);
    // printf("ref count of self: %i\nref count of array:
    // %i\n\n",self->ob_base.ob_refcnt,self->array->ob_refcnt);
    // if(PyArray_SetBaseObject((PyArrayObject*)self->array,(PyObject*)self)<0){
    // printf("Failed to set buffer for array.\n");
    // return -1;
    //}
    // printf("ref count of self: %i\nref count of array:
    // %i\n\n",self->ob_base.ob_refcnt,self->array->ob_refcnt);


    return 0;
}

static PyObject* create(Noddy* self)
{
    npy_intp dims[] = { 10 };
    PyObject* array = PyArray_SimpleNewFromData(1, dims, NPY_INT32, self->data);
    Py_INCREF(self);
    if (PyArray_SetBaseObject((PyArrayObject*)array, (PyObject*)self) < 0) {
        printf("Failed to set buffer for array\n");
        return Py_None;
    }
    PyArray_CLEARFLAGS((PyArrayObject*)array, NPY_ARRAY_WRITEABLE);
    return array;
}


// static PyMemberDef Noddy_members[] = {
//    {"int", T_OBJECT_EX, offsetof(Noddy, ), 0,
//     "the array"},
//    {NULL}  /* Sentinel */
//};

static PyMethodDef Noddy_methods[] = { { "create", (PyCFunction)create, METH_NOARGS,
                                         "create array" },
                                       { NULL } };


static PyTypeObject NoddyType = {
    PyVarObject_HEAD_INIT(NULL, 0) "testType.Noddy", /* tp_name */
    sizeof(Noddy), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)Noddy_dealloc, /* tp_dealloc */
    0, /* tp_print */
    0, /* tp_getattr */
    0, /* tp_setattr */
    0, /* tp_reserved */
    0, /* tp_repr */
    0, /* tp_as_number */
    0, /* tp_as_sequence */
    0, /* tp_as_mapping */
    0, /* tp_hash  */
    0, /* tp_call */
    0, /* tp_str */
    0, /* tp_getattro */
    0, /* tp_setattro */
    0, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    "Noddy objects", /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    0, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    0, /* tp_iter */
    0, /* tp_iternext */
    Noddy_methods, /* tp_methods */
    0, // Noddy_members,             /* tp_members */
    0, /* tp_getset */
    0, /* tp_base */
    0, /* tp_dict */
    0, /* tp_descr_get */
    0, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc)Noddy_init, /* tp_init */
    0, /* tp_alloc */
    Noddy_new, /* tp_new */
};

static PyModuleDef testTypemodule =
{ PyModuleDef_HEAD_INIT, "testType", "Example module that creates an extension type.", -1, NULL, NULL, NULL, NULL, NULL };

PyMODINIT_FUNC PyInit_testType(void)
{
    PyObject* m;

    import_array();

    if (PyType_Ready(&NoddyType) < 0) return NULL;

    m = PyModule_Create(&testTypemodule);
    if (m == NULL) return NULL;

    Py_INCREF(&NoddyType);
    PyModule_AddObject(m, "Noddy", (PyObject*)&NoddyType);
    return m;
}
