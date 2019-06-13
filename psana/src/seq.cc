
#include "xtcdata/xtc/Sequence.hh"

#include <Python.h>
#include <structmember.h>
#include <stdio.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <assert.h>

using namespace std;
using namespace XtcData;

typedef struct {
    PyObject_HEAD
    Sequence* seq;
} PySeqObject;

static void seq_dealloc(PySeqObject* self)
{
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* seq_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    PySeqObject* self;
    self = (PySeqObject*)type->tp_alloc(type, 0);
    return (PyObject*)self;
}

static int seq_init(PySeqObject* self, PyObject* args, PyObject* kwds)
{
    PyObject* seqptr_capsule;
    PyArg_ParseTuple(args, "O", &seqptr_capsule);
    self->seq = (Sequence*)PyCapsule_GetPointer(seqptr_capsule, NULL);
    return 0;
}

static PyMemberDef seq_members[] = {
    { NULL }
};

static PyObject* service(PySeqObject* self) {
    return PyLong_FromLong(self->seq->service());
}

static PyObject* timestamp(PySeqObject* self) {
  return PyLong_FromLong(self->seq->stamp().value());
}

static PyObject* pulseid(PySeqObject* self) {
    return PyLong_FromLong(self->seq->pulseId().value());
}

static PyMethodDef seq_methods[] = {
    {"service", (PyCFunction)service, METH_NOARGS, "service"},
    {"timestamp", (PyCFunction)timestamp, METH_NOARGS, "timestamp"},
    {"pulseid", (PyCFunction)pulseid, METH_NOARGS, "pulseid"},
    {NULL}  /* Sentinel */
};

static PyTypeObject seq_SeqType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "seq.Seq", /* tp_name */
    sizeof(PySeqObject), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)seq_dealloc, /* tp_dealloc */
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
    seq_methods, /* tp_methods */
    seq_members, /* tp_members */
    0, /* tp_getset */
    0, /* tp_base */
    0, /* tp_dict */
    0, /* tp_descr_get */
    0, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc)seq_init, /* tp_init */
    0, /* tp_alloc */
    seq_new, /* tp_new */
    0, /* tp_free;  Low-level free-memory routine */
    0, /* tp_is_gc;  For PyObject_IS_GC */
    0, /* tp_bases*/
    0, /* tp_mro;  method resolution order */
    0, /* tp_cache*/
    0, /* tp_subclasses*/
    0, /* tp_weaklist*/
    (destructor)seq_dealloc, /* tp_del*/
};

#if PY_MAJOR_VERSION > 2
static PyModuleDef seqmodule =
{ PyModuleDef_HEAD_INIT, "seq", NULL, -1, NULL, NULL, NULL, NULL, NULL };
#endif

#ifndef PyMODINIT_FUNC /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

#if PY_MAJOR_VERSION > 2
PyMODINIT_FUNC PyInit_seq(void)
{
    PyObject* m;

    if (PyType_Ready(&seq_SeqType) < 0) {
        return NULL;
    }

    m = PyModule_Create(&seqmodule);
    if (m == NULL) {
        return NULL;
    }

    Py_INCREF(&seq_SeqType);
    PyModule_AddObject(m, "Seq", (PyObject*)&seq_SeqType);
    return m;
}
#else
PyMODINIT_FUNC initseq(void) {
    PyObject *m;
    
    if (PyType_Ready(&seq_SeqType) < 0)
        return;

    m = Py_InitModule3("seq", seq_methods, "Seq module.");

    if (m == NULL)
        return;

    Py_INCREF(&seq_SeqType);
    PyModule_AddObject(m, "Seq", (PyObject *)&seq_SeqType);
}
#endif
