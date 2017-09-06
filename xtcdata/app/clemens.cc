#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>

#include <structmember.h>



typedef struct {

    PyObject_HEAD

    PyObject* dict;

    int* dgram;

} DgramObject;



static void

dgram_dealloc(DgramObject* self)

{

    printf("dgram_dealloc\n");

    Py_XDECREF(self->dict);

    delete [] self->dgram;

    Py_TYPE(self)->tp_free((PyObject*)self);

}



static PyObject *

dgram_new(PyTypeObject* type, PyObject* args, PyObject* kwds)

{

    DgramObject* self = (DgramObject*)type->tp_alloc(type, 0);

    if (self == NULL)

        return NULL;



    self->dict = PyDict_New();

    if(self->dict == NULL) {

        Py_DECREF(self);

        return NULL;

    }

    return (PyObject*)self;

}



static int

dgram_init(DgramObject* self, PyObject* args, PyObject* kwds)

{

    self->dgram = new int[10];

    for (int i = 0; i<10; i++){

        self->dgram[i] = i;

    }

    /*
 *
 *     npy_intp dims[] = {10};
 *
 *         PyObject* array = PyArray_SimpleNewFromData(1, dims, NPY_INT32, self->dgram);
 *
 *             Py_INCREF(self);
 *
 *                 if (PyArray_SetBaseObject((PyArrayObject*)array, (PyObject*)self) < 0) {
 *
 *                         Py_DECREF(self);
 *
 *                                 return NULL;
 *
 *                                     } 
 *
 *                                         
 *
 *                                             if(PyObject_SetAttrString((PyObject*)self, "array", array) < 0) {
 *
 *                                                     Py_DECREF(self);
 *
 *                                                             return NULL;
 *
 *                                                                 }
 *
 *                                                                     Py_DECREF(array);
 *
 *                                                                         */

}



static PyObject*

dgram_get_item(DgramObject* self, PyObject* key)

{

    npy_intp dims[] = {10};

    PyObject* array = PyArray_SimpleNewFromData(1, dims, NPY_INT32, self->dgram);

    Py_INCREF(self);

    if (PyArray_SetBaseObject((PyArrayObject*)array, (PyObject*)self) < 0) {

        Py_DECREF(self);

        return NULL;

    }

    return array;

}



static PyMappingMethods dgram_as_mapping = {

    (lenfunc)NULL,         /* inquiry mp_length  * __len__ */

    (binaryfunc)dgram_get_item,     /* binaryfunc mp_subscript  * __getitem__ */

    (objobjargproc)NULL,     /* objobjargproc mp_ass_subscript  * __setitem__ */

};





static PyMemberDef dgram_members[] = {

    {"__dict__", T_OBJECT_EX, offsetof(DgramObject,dict)}, {0}

};



static PyTypeObject Dgram_Type = {

    PyVarObject_HEAD_INIT(NULL, 0)

    "dgram.Dgram",             /* tp_name */

    sizeof(DgramObject), /* tp_basicsize */

    0,                         /* tp_itemsize */

    (destructor)dgram_dealloc, /* tp_dealloc */

    0,                         /* tp_print */

    0,                         /* tp_getattr */

    0,                         /* tp_setattr */

    0,                         /* tp_compare */

    0,                         /* tp_repr */

    0,                         /* tp_as_number */

    0,                         /* tp_as_sequence */

    &dgram_as_mapping,     /* tp_as_mapping */

    0,                         /* tp_hash */

    0,                         /* tp_call */

    0,                         /* tp_str */

    PyObject_GenericGetAttr,   /* tp_getattro */

    0,                         /* tp_setattro */

    0,                         /* tp_as_buffer */

    Py_TPFLAGS_DEFAULT,        /* tp_flags */

    0,                       /* tp_doc */

    0,                       /* tp_traverse */

    0,                       /* tp_clear */

    0,                       /* tp_richcompare */

    0,                       /* tp_weaklistoffset */

    0,                       /* tp_iter */

    0,                       /* tp_iternext */

    0,                        /* tp_methods */

    dgram_members,             /* tp_members */

    0,                       /* tp_getset */

    0,                       /* tp_base */

    0,                       /* tp_dict */

    0,                       /* tp_descr_get */

    0,                       /* tp_descr_set */

    offsetof(DgramObject,dict),        /* tp_dictoffset */

    (initproc)dgram_init,                   /* tp_init */

    0,                       /* tp_alloc */

    dgram_new,                       /* tp_new */

    0,                  /*tp_free;  Low-level free-memory routine */

    0,                   /*tp_is_gc;  For PyObject_IS_GC */

    0,                  /*tp_bases*/

    0,                /*tp_mro;  method resolution order */

    0,                 /*tp_cache*/

    0,                  /*tp_subclasses*/

    0,                  /*tp_weaklist*/

    (destructor)dgram_dealloc,                /*tp_del*/

};



static PyModuleDef dgrammodule = {

    PyModuleDef_HEAD_INIT,

    "dgram",

    "Example module that creates an extension type.",

    -1,

    NULL, NULL, NULL, NULL, NULL

};



PyMODINIT_FUNC

PyInit_dgram(void)

{

    import_array();



    if (PyType_Ready(&Dgram_Type) < 0){

        return NULL;

    }



    PyObject* m = PyModule_Create(&dgrammodule);

    if (m == NULL) {

        return NULL;

    }



    Py_INCREF(&Dgram_Type);

    PyModule_AddObject(m, "Dgram", (PyObject *)&Dgram_Type);

    return m;

}
