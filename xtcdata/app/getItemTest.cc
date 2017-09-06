#include "pdsdata/xtc/Descriptor.hh"
#include "pdsdata/xtc/Dgram.hh"
#include "pdsdata/xtc/TypeId.hh"
#include "pdsdata/xtc/XtcIterator.hh"
#include "pdsdata/xtc/DetInfo.hh"
#include "pdsdata/xtc/ProcInfo.hh"
#include "pdsdata/xtc/XtcFileIterator.hh"


#include "pdsdata/xtc/Hdf5Writer.hh"
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <unistd.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <structmember.h>

#include <fcntl.h>






using namespace Pds;
#define BUFSIZE 0x4000000

typedef struct {
    PyObject_HEAD
    PyObject* dict;
    Dgram* dgram;
    Data* data;
} DgramObject;
//dataDict_DgramObject;

PyObject* DataAssign(DgramObject* dgram, Data& d, Field& f)
{  
    const char *tempName = f.name;
    PyObject *value;
    if (f.rank == 0)  
    {
      switch(f.type){
        case UINT8:{
        const int tempVal = d.get_value<uint8_t>(tempName);
        value = Py_BuildValue("i",tempVal);
        break;
        } 
        case UINT16:{
        const int tempVal = d.get_value<uint16_t>(tempName);
        value = Py_BuildValue("i",tempVal);
        break;
        }
        case INT32:{
          const int tempVal = d.get_value<int32_t>(tempName);
          value = Py_BuildValue("i",tempVal);
          break;
        }
        case FLOAT:{
          const float tempVal = d.get_value<float>(tempName);
          value = Py_BuildValue("f",tempVal);
          break;
        }
        case DOUBLE:{
          const int tempVal = d.get_value<double>(tempName);
          value = Py_BuildValue("d",tempVal);
          break;
        }
      }
    }
    else
    {
      npy_intp dims[f.rank];
      for (int i = 0; i<f.rank; i++){
        dims[i] = f.shape[i];
      }
      switch(f.type){
        case UINT8:{
          value = PyArray_SimpleNewFromData(f.rank,dims,NPY_UINT8,d.get_buffer()+f.offset);
          break;
        }
        case UINT16:{
          value = PyArray_SimpleNewFromData(f.rank,dims,NPY_UINT16,d.get_buffer()+f.offset);
          break;
        }
        case INT32:{
          value = PyArray_SimpleNewFromData(f.rank,dims,NPY_INT32,d.get_buffer()+f.offset);
          break;
        }
        case FLOAT:{
          value = PyArray_SimpleNewFromData(f.rank,dims,NPY_FLOAT,d.get_buffer()+f.offset);
          break;
        }
        case DOUBLE:{
          value = PyArray_SimpleNewFromData(f.rank,dims,NPY_DOUBLE,d.get_buffer()+f.offset);
          break;
        }
      }

      Py_INCREF(dgram);

      if (PyArray_SetBaseObject((PyArrayObject*)value, (PyObject*)dgram) < 0) {

        printf("Failed to set buffer for array.\n");

        Py_DECREF(dgram);

        return NULL;

      }

   } 
   Py_DECREF(value);
}


// this represent the "analysis" code
class myLevelIter : public XtcIterator
{
    public:
    enum { Stop, Continue };
    myLevelIter(Xtc* xtc, unsigned depth, DgramObject* dgram) : XtcIterator(xtc), _depth(depth), _dgram(dgram)
    {

    }

    int process(Xtc* xtc)
    {
        unsigned i = _depth;
        switch (xtc->contains.id()) {
        case (TypeId::Parent): {
            printf("THe parent is getting called.\n\n");
            myLevelIter iter(xtc, _depth + 1, _dgram);
            iter.iterate();
            break;
        }
        case (TypeId::Data): {
            //_dgram->data = (Data*)xtc->payload();
            Data& d = *(Data*)xtc->payload();
            printf("(Data*)xtc->payload(): %p\nsecond time %p\n",*(Data*)xtc->payload(),*(Data*)xtc->payload());
            Descriptor& desc = d.desc();
            printf("Found fields named:\n");

            //std::cout << d.get_value<float>("myfloat") << std::endl;

            //auto array = d.get_value<Array<float>>("array");
            //for (int i = 0; i < 3; i++) {
            //    for (int j = 0; j < 3; j++) {
            //        std::cout << array(i, j) << "  ";
            //    }
            //    std::cout << std::endl;
            //}

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
    DgramObject* _dgram;
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
dgram_dealloc(DgramObject *self)
{
  printf("Here in dealloc\n");
  Py_XDECREF(self->dict);
  Py_TYPE(self)->tp_free((PyObject*)self);
  free(self->dgram);
}

static PyObject *
dgram_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  DgramObject *self;

  self = (DgramObject *)type->tp_alloc(type,0);
  
  
  if (self != NULL) {
    self->dict = PyDict_New();
  }  

  return (PyObject *) self;
}

static int
dgram_init(DgramObject *self, PyObject *args, PyObject *kwds)
{
    printf("here in dgram_init %p %d\n", self, Py_REFCNT(self));

    self->dgram = (Dgram*)malloc(BUFSIZE);

    int fd = open("data.xtc", O_RDONLY | O_LARGEFILE);

    if(::read(fd,self->dgram,sizeof(*self->dgram))==0){ 
      printf("read was unsuccessful.\n");
      return 0;
    }
    
    size_t payloadSize = self->dgram->xtc.sizeofPayload();
    size_t sz = ::read(fd,self->dgram->xtc.payload(),payloadSize);

    printf("Xtc pointer from init: %p\n\n",&self->dgram->xtc);

    close(fd);


    myLevelIter iter(&self->dgram->xtc, 0, self);
    iter.iterate();
    //

    return 0;
}



static PyObject*

dgram_get_item(DgramObject* self, PyObject* key)

{
    //Here we acquire the field which we will then use to
    //classify the data that "key" points to
    Xtc* xtc = &self->dgram->xtc;
    printf("Xtc pointer from get_item func: %p\n",xtc);

    printf("Here is the contains ID output: %i\n\n", xtc->contains.id());




    Data* d = (Data*)xtc->payload();
    printf("Data pointer from get_item func: %p\n",*d);
    Descriptor& desc = d->desc();

    printf("xtc: %p\ndata: %p\ndesc: %p\n",xtc,d,desc);


    const char* name;

    if(PyUnicode_READY(key) == 0){
        Py_UCS1* pyNamePtr = PyUnicode_1BYTE_DATA(key);
        name = (char*)pyNamePtr;
    }
        

    printf("Here is the result of the parse: %s\n\n",name);
    printf("here is desc.num_fields: %i\n\n", desc.num_fields);

    for(int i = 0; i < desc.num_fields; i++){
        printf("Made it past the field assign\n");
        Field& f = desc.get(i);
        printf("Name of field: %s\n",f.name);
    }


    //Field& f = *desc.get_field_by_name(name);

    

    PyObject* result = NULL;// = DataAssign(self,d,f);

    return result;

}



static PyMappingMethods dgram_as_mapping = {

    (lenfunc)NULL,         /* inquiry mp_length  * __len__ */

    (binaryfunc)dgram_get_item,     /* binaryfunc mp_subscript  * __getitem__ */

    (objobjargproc)NULL,     /* objobjargproc mp_ass_subscript  * __setitem__ */

};



static PyMemberDef dgram_members[] = {
    {"__dict__", T_OBJECT_EX, offsetof(DgramObject,dict), 0,
     "attribute dictionary"},
    {NULL}
};

//PyObject * tp_getattr(PyObject* o, char* key)
//{
//  printf("\n\nIt just called the getattr function \n\n");
//  return Py_None;
//}

 
PyObject * tp_getattro(PyObject* o, PyObject* key)
{
  printf("\n\nIt just called the getattro function \n\n");
  PyObject* res = PyDict_GetItem(((DgramObject*) o) -> dict,key);
  if(res != NULL){
    Py_INCREF(res);
    if(*res->ob_type->tp_name == *"numpy.ndarray"){
      printf("we're in the if\n\n");
      Py_INCREF(o);  
    }
  }
  else{
    printf("We're in the else\n\n");
    res = PyObject_GenericGetAttr(o,key);
  }

  return res;
}



static PyTypeObject DgramType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "dgram.Dgram",             /* tp_name */
  sizeof(DgramObject), /* tp_basicsize */
  0,                         /* tp_itemsize */
  (destructor)dgram_dealloc, /* tp_dealloc */
  0,                         /* tp_print */
  0,//tp_getattr,                         /* tp_getattr */
  0,                         /* tp_setattr */
  0,                         /* tp_compare */
  0,                         /* tp_repr */
  0,                         /* tp_as_number */
  0,                         /* tp_as_sequence */
  &dgram_as_mapping,                         /* tp_as_mapping */
  0,                         /* tp_hash */
  0,                         /* tp_call */
  0,                         /* tp_str */
  PyObject_GenericGetAttr,// tp_getattro,                        /* tp_getattro */
  0,                         /* tp_setattro */
  0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,// | Py_TPFLAGS_BASETYPE, /* tp_flags */
  0,                       /* tp_doc */
  0,                       /* tp_traverse */
  0,                       /* tp_clear */
  0,                       /* tp_richcompare */
  0,                       /* tp_weaklistoffset */
  0,                       /* tp_iter */
  0,                       /* tp_iternext */
  0,          /* tp_methods */
  dgram_members,                       /* tp_members */
  0,                       /* tp_getset */
  0,                       /* tp_base */
  0,                       /* tp_dict */
  0,                       /* tp_descr_get */
  0,                       /* tp_descr_set */
  offsetof(DgramObject,dict),                       /* tp_dictoffset */
  (initproc)dgram_init,   /* tp_init */
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


static PyModuleDef dictmodule = {
    PyModuleDef_HEAD_INIT,
    "getItem",
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
PyInit_getItem(void)
{
  PyObject* m;

  import_array();

  //dgram_DgramType.tp_new = PyType_GenericNew;
  
  if (PyType_Ready(&DgramType) < 0){
    return NULL;
  }

  m = PyModule_Create(&dictmodule);
  if (m == NULL){
    return NULL;
  }

//  m = Py_InitModule3("dgram", NULL, "Example dgram module");

  Py_INCREF(&DgramType);
  PyModule_AddObject(m, "Dgram", (PyObject *)&DgramType);
  return m;
}


