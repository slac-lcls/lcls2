#include "pdsdata/xtc/Descriptor.hh"
#include "pdsdata/xtc/Dgram.hh"
#include "pdsdata/xtc/TypeId.hh"
#include "pdsdata/xtc/XtcIterator.hh"

#include "pdsdata/xtc/Hdf5Writer.hh"
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <unistd.h>
#include <Python.h>
#include <numpy/arrayobject.h>

using namespace Pds;
#define BUFSIZE 0x4000000

typedef struct {
    PyDictObject dict;
} dgram_DgramObject;
//dataDict_DgramObject;

// this represent the "analysis" code
class myLevelIter : public XtcIterator
{
    public:
    enum { Stop, Continue };
    myLevelIter(Xtc* xtc, unsigned depth, PyObject* dictionary) : XtcIterator(xtc), _depth(depth), _dictionary(dictionary)
    {

    }

    int process(Xtc* xtc)
    {
        unsigned i = _depth;
        switch (xtc->contains.id()) {
        case (TypeId::Parent): {
            myLevelIter iter(xtc, _depth + 1, _dictionary);
            iter.iterate();
            break;
        }
        case (TypeId::Data): {
            Data& d = *(Data*)xtc->payload();
            Descriptor& desc = d.desc();
            printf("Found fields named:\n");
            for (int i = 0; i < desc.num_fields; i++) {
                Field& f = desc.get(i);
                printf("%s  offset: %d\n", f.name, f.offset);
                if (f.type == FLOAT){
                    const char *tempName = f.name;
                    const float tempVal = d.get_value<float>(tempName);
                    PyObject *key = PyUnicode_FromString(tempName);
                    PyObject *value = Py_BuildValue("f",tempVal);
                    PyDict_SetItem(_dictionary,key,value);
                }
            }

            std::cout << d.get_value<float>("myfloat") << std::endl;

            auto array = d.get_value<Array<float>>("array");
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    std::cout << array(i, j) << "  ";
                }
                std::cout << std::endl;
            }

            break;
        }
        default:
            printf("TypeId %s (value = %d)\n", TypeId::name(xtc->contains.id()), (int)xtc->contains.id());
            break;
        }
        return Continue;
    }

    private:
    unsigned _depth;
    PyObject* _dictionary;
};

// everything below here is inletWire code

class MyData : public Data
{
    public:
    void* operator new(size_t size, void* p)
    {
        return p;
    }
    MyData(float f, int i1, int i2) : Data(sizeof(*this))
    {
        _fdata = f;
        _idata = i1;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                array[i][j] = i * j;
            }
        }
    }

    private:
    float _fdata;
    float array[3][3];
    int _idata;
};

static void
dgram_dealloc(dgram_DgramObject *self)
{
  printf("here in dealloc\n");
}

static int
dgram_init(dgram_DgramObject *self, PyObject *args, PyObject *kwds)
{
    printf("here in dgram_init\n");
    if (PyDict_Type.tp_init((PyObject *)self, args, kwds) < 0) {
      printf("whoops\n");
      return -1;
    }

    // this is the datagram, which gives you an "xtc" for free
    Dgram& dgram = *(Dgram*)malloc(BUFSIZE);
    TypeId tid(TypeId::Parent, 0);
    dgram.xtc.contains = tid;
    dgram.xtc.damage = 0;
    dgram.xtc.extent = sizeof(Xtc);

    // make a child xtc with detector data and descriptor
    TypeId tid_child(TypeId::Data, 0);
    Xtc& xtcChild = *new (&dgram.xtc) Xtc(tid_child);

    // creation of fixed-length data in xtc
    MyData& d = *new (xtcChild.alloc(sizeof(MyData))) MyData(1, 2, 3);

    // creation of variable-length data in xtc
    DescriptorManager descMgr(xtcChild.next());
    descMgr.add("myfloat", FLOAT);

    int shape[] = {3, 3};
    descMgr.add("array", FLOAT, 2, shape);

    descMgr.add("myint", INT32);

    xtcChild.alloc(descMgr.size());

    // update parent xtc with our new size.
    dgram.xtc.alloc(xtcChild.sizeofPayload());

    HDF5File file("test.h5");
    file.addDatasets(descMgr._desc);
    file.appendData(d);

    myLevelIter iter(&dgram.xtc, 0, (PyObject *) self);
    iter.iterate();

    free((void*)&dgram);

    return 0;
}

//static PyMethodDef dictMethods[] = {
//    {"dictInit", (PyCFunction)dataDict_dictInit, METH_NOARGS, "Initialize a simple dictionary."},
//
//    {NULL}
//};

static PyTypeObject dgram_DgramType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "dgram.Dgram",             /* tp_name */
  sizeof(dgram_DgramObject), /* tp_basicsize */
  0,                         /* tp_itemsize */
  (destructor)dgram_dealloc, /* tp_dealloc */
  0,                         /* tp_print */
  0,                         /* tp_getattr */
  0,                         /* tp_setattr */
  0,                         /* tp_compare */
  0,                         /* tp_repr */
  0,                         /* tp_as_number */
  0,                         /* tp_as_sequence */
  0,                         /* tp_as_mapping */
  0,                         /* tp_hash */
  0,                         /* tp_call */
  0,                         /* tp_str */
  0,                         /* tp_getattro */
  0,                         /* tp_setattro */
  0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
  Py_TPFLAGS_BASETYPE, /* tp_flags */
  0,                       /* tp_doc */
  0,                       /* tp_traverse */
  0,                       /* tp_clear */
  0,                       /* tp_richcompare */
  0,                       /* tp_weaklistoffset */
  0,                       /* tp_iter */
  0,                       /* tp_iternext */
  0,          /* tp_methods */
  0,                       /* tp_members */
  0,                       /* tp_getset */
  0,                       /* tp_base */
  0,                       /* tp_dict */
  0,                       /* tp_descr_get */
  0,                       /* tp_descr_set */
  0,                       /* tp_dictoffset */
  (initproc)dgram_init,   /* tp_init */
  0,                       /* tp_alloc */
  0,                       /* tp_new */
};


static struct PyModuleDef dictmodule = {
    PyModuleDef_HEAD_INIT,
    "dgram",
    NULL,
    -1,
    NULL,NULL,NULL,NULL,NULL
};

/*
PyMODINIT_FUNC
PyInit_dgram(void)
{
    import_array();
    return PyModule_Create(&dictmodule);
}
*/

#ifndef PyMODINIT_FUNC/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
PyInit_dgram(void)
{
  PyObject* m;

  dgram_DgramType.tp_base = &PyDict_Type;

  if (PyType_Ready(&dgram_DgramType) < 0)
    return NULL;

  m = PyModule_Create(&dictmodule);
  if (m == NULL)
    return NULL;

//  m = Py_InitModule3("dgram", NULL, "Example dgram module");

  Py_INCREF(&dgram_DgramType);
  PyModule_AddObject(m, "Dgram", (PyObject *)&dgram_DgramType);
  import_array();
  return m;
}


