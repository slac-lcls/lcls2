#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/DescData.hh"

#include "xtcdata/xtc/TypeId.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/NamesIter.hh"

#include <Python.h>
#include <stdio.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <structmember.h>

using namespace XtcData;
#define TMPSTRINGSIZE 1024

static const char* PyNameDelim=".";
static const char EnumDelim=':';

using namespace std;

#define SLEEP_SECS 1

// to avoid compiler warnings for debug variables
#define _unused(x) ((void)(x))

struct ContainerInfo {
    PyObject* containermod;
    PyObject* pycontainertype;
};

struct PyDgramObject {
    PyObject_HEAD
    PyObject* dict;
    PyObject* dgrambytes;
    Dgram* dgram;
    int file_descriptor;
    ssize_t offset;
    Py_buffer buf;
    ContainerInfo contInfo;
    NamesIter* namesIter;   // only nonzero in the config dgram
    ssize_t size;           // size of dgram - for allocating dgram of any size
    int max_retries;        // set no. of retries when reading data (default=0)
};

static void addObjToPyObj(PyObject* parent, const char* name, PyObject* obj, PyObject* pycontainertype) {
    char namecopy[TMPSTRINGSIZE];
    strncpy(namecopy,name,TMPSTRINGSIZE-1);
    char *key = ::strtok(namecopy,PyNameDelim);
    while(1) {
        char* next = ::strtok(NULL, PyNameDelim);
        bool last = (next == NULL);
        if (last) {
            // add the real object
            int fail = PyObject_SetAttrString(parent, key, obj);
            if (fail) printf("addObj: failed to set object attribute\n");
            Py_DECREF(obj); // transfer ownership to parent
            break;
        } else {
            if (!PyObject_HasAttrString(parent, key)) {
                PyObject* container = PyObject_CallObject(pycontainertype, NULL);
                int fail = PyObject_SetAttrString(parent, key, container);
                if (fail) printf("addObj: failed to set container attribute\n");
                Py_DECREF(container); // transfer ownership to parent
            }
            parent = PyObject_GetAttrString(parent, key);
            Py_DECREF(parent); // we just want the pointer, not a new reference
        }
        key=next;
    }
}

// main routine to add hierarchical structures to python objects.
// sets parent.detname[segment].attr1.attr2.attr3 = obj
// where name is a delimited string formatted like "detname_attr1_attr2_attr3"
static void addObjHierarchy(PyObject* parent, PyObject* pycontainertype,
                            const char* name, PyObject* obj,
                            unsigned segment) {
    char namecopy[TMPSTRINGSIZE];
    strncpy(namecopy,name,TMPSTRINGSIZE-1);
    char *key = ::strtok(namecopy,PyNameDelim);
    char* next = ::strtok(NULL, PyNameDelim);

    PyObject* dict;
    if (!PyObject_HasAttrString(parent, key)) {
        dict = PyDict_New();
        int fail = PyObject_SetAttrString(parent, key, dict);
        if (fail) printf("Dgram: failed to set container attribute\n");
    } else {
        dict = PyObject_GetAttrString(parent, key);
    }
    // either way we got a new reference to the dict.
    // keep the parent as the owner.
    Py_DECREF(dict); // transfer ownership to parent

    bool last = (next == NULL);
    PyObject* pySeg = Py_BuildValue("i",segment);
    if (last) {
        // we're at the lowest level, set the value to be the data object.
        // this case should happen rarely, if ever, in lcls2.
        PyDict_SetItem(dict,pySeg,obj);
        Py_DECREF(obj); // transfer ownership to parent
    } else {
        // we're not at the lowest level, get the container object for this segment
        PyObject* container;
        if (!(container=PyDict_GetItem(dict,pySeg))) {
            container = PyObject_CallObject(pycontainertype, NULL);
            PyDict_SetItem(dict,pySeg,container);
            Py_DECREF(container); // transfer ownership to parent
        }
        // add the rest of the fields to the container.  note
        // that we compute the offset in the original string,
        // to exclude the detname that we have processed above,
        // since strtok has messed with our copy of the original string.
        addObjToPyObj(container,name+(next-key),obj,pycontainertype);
    }
}

// add _xtc (_ prefix hides this from the detector interface)
// and its attributes to dgram object
static void setXtc(PyObject* parent, PyObject* pycontainertype, Xtc* myXtc) {

    int fail = 0;
    PyObject* pyXtc;
    pyXtc = PyObject_CallObject(pycontainertype, NULL);
    fail = PyObject_SetAttrString(parent, "_xtc", pyXtc);
    if (fail) throw "setXtc: failed to set container _xtc attribute\n";
    Py_DECREF(pyXtc);

    uint16_t damage = myXtc->damage.value();
    PyObject* pyDamage = Py_BuildValue("H", damage);
    fail = PyObject_SetAttrString(pyXtc, "damage", pyDamage);
    if (fail) throw "setXtc: failed to set container damage attribute\n";
    Py_DECREF(pyDamage);
}

// add _xtc to each segment of det_name dictionary object
static void setXtcForSegment(PyObject* parent, PyObject* pycontainertype,
        const char* detName, unsigned segment, Xtc* myXtc) {
    PyObject* dict;
    dict = PyObject_GetAttrString(parent, detName);
    Py_DECREF(dict); // transfer ownership to parent

    // get segment container for this segment
    PyObject* pySeg = Py_BuildValue("i", segment);
    PyObject* container;
    container = PyDict_GetItem(dict, pySeg);
    Py_DECREF(pySeg);

    const char* xtcLabel = "_xtc";
    int hasXtcLabel = PyObject_HasAttrString(container, xtcLabel);
    if (hasXtcLabel == 1) return;

    setXtc(container, pycontainertype, myXtc);
}

static void setAlg(PyObject* parent, PyObject* pycontainertype, const char* baseName, Alg& alg, unsigned segment) {
    const char* algName = alg.name();
    const uint32_t _v = alg.version();
    char keyName[TMPSTRINGSIZE];

    PyObject* software = Py_BuildValue("s", algName);
    PyObject* version  = Py_BuildValue("iii", (_v>>16)&0xff, (_v>>8)&0xff, (_v)&0xff);

    snprintf(keyName,sizeof(keyName),"%s%ssoftware",
             baseName,PyNameDelim);
    addObjHierarchy(parent, pycontainertype, keyName, software, segment);
    snprintf(keyName,sizeof(keyName),"%s%sversion",
             baseName,PyNameDelim);
    addObjHierarchy(parent, pycontainertype, keyName, version, segment);
}

static void setDataInfo(PyObject* parent, PyObject* pycontainertype, const char* baseName, Name& name, unsigned segment) {
    unsigned type = name.type();
    unsigned rank = name.rank();
    char keyName[2*TMPSTRINGSIZE];

    PyObject* py_type = Py_BuildValue("i", type);
    PyObject* py_rank = Py_BuildValue("i", rank);

    snprintf(keyName,sizeof(keyName),"%s%s_type",
             baseName,PyNameDelim);
    addObjHierarchy(parent, pycontainertype, keyName, py_type, segment);
    snprintf(keyName,sizeof(keyName),"%s%s_rank",
             baseName,PyNameDelim);
    addObjHierarchy(parent, pycontainertype, keyName, py_rank, segment);
}

static void setDetInfo(PyObject* parent, PyObject* pycontainertype, Names& names) {
    char keyName[2*TMPSTRINGSIZE];
    unsigned segment = names.segment();
    PyObject* detType = Py_BuildValue("s", names.detType());
    snprintf(keyName,sizeof(keyName),"%s%sdettype",
             names.detName(),PyNameDelim);
    addObjHierarchy(parent, pycontainertype, keyName, detType, segment);

    PyObject* detId = Py_BuildValue("s", names.detId());
    snprintf(keyName,sizeof(keyName),"%s%sdetid",
             names.detName(),PyNameDelim);
    addObjHierarchy(parent, pycontainertype, keyName, detId, segment);
}

static void dictAssignConfig(PyDgramObject* pyDgram, NamesLookup& namesLookup)
{
    // This function gets called at configure: add attributes "software" and "version" to pyDgram and return
    char baseName[TMPSTRINGSIZE];
    PyObject* pycontainertype = pyDgram->contInfo.pycontainertype;

    PyObject* software;
    if (!PyObject_HasAttrString((PyObject*)pyDgram, "software")) {
        software = PyObject_CallObject(pycontainertype, NULL);
        int fail = PyObject_SetAttrString((PyObject*)pyDgram, "software", software);
        if (fail) throw "dictAssignConfig: failed to set container attribute\n";
        Py_DECREF(software); // transfer ownership to parent
    } else {
        throw "dictAssignConfig: software attribute already exists\n";
    }

    for (auto & namesPair : namesLookup) {
        NameIndex& nameIndex = namesPair.second;
        if (!nameIndex.exists()) continue;
        Names& names = nameIndex.names();
        Alg& detAlg = names.alg();
        unsigned segment = names.segment();
        snprintf(baseName,sizeof(baseName),"%s%s%s",
                 names.detName(),PyNameDelim,names.alg().name());
        setAlg(software, pycontainertype, baseName, detAlg, segment);
        setDetInfo(software, pycontainertype, names);

        for (unsigned j = 0; j < names.num(); j++) {
            Name& name = names.get(j);
            Alg& alg = name.alg();
            snprintf(baseName,sizeof(baseName),"%s%s%s%s%s",
                     names.detName(),PyNameDelim,names.alg().name(),
                     PyNameDelim,name.name());
            setAlg(software, pycontainertype, baseName, alg, segment);
            setDataInfo(software, pycontainertype, baseName, name, segment);
        }
    }
}

// return an "enum object" (with a value/dict that can be added to the pydgram
static PyObject* createEnum(const char* enumname, PyDgramObject* pyDgram, DescData& descdata) {
    char tempName[TMPSTRINGSIZE];
    const char* enumtype = strchr(enumname,EnumDelim)+1;
    Names& names = descdata.nameindex().names();

    // make a container
    PyObject* parent = PyObject_CallObject(pyDgram->contInfo.pycontainertype, NULL);
    // add the dict associated with the enum to the container
    // fill in the dict and the value
    PyObject* dict = PyDict_New();
    PyObject_SetAttrString(parent, "names", dict);
    Py_DECREF(dict); // transfer ownership to parent

    for (unsigned i = 0; i < names.num(); i++) {
        Name& name = names.get(i);
        const char* varName = name.name();

        if (name.type() == Name::ENUMVAL) {
            if (strncmp(enumname,varName,TMPSTRINGSIZE)==0) {
                const auto tempVal = descdata.get_value<uint32_t>(varName);
                PyObject* newobj = Py_BuildValue("i", tempVal);
                PyObject_SetAttrString(parent, "value", newobj);
            }
        } else if (name.type() == Name::ENUMDICT) {
            // check that we are looking at the correct ENUMDICT
            strncpy(tempName,varName,TMPSTRINGSIZE-1);
            char* enumtype_dict = strchr(tempName,EnumDelim)+1;

            if (strncmp(enumtype,enumtype_dict,TMPSTRINGSIZE)==0) {
                // eliminate the enum type from the name by
                // inserting the null character
                enumtype_dict[-1]='\0';
                const auto tempVal = descdata.get_value<uint32_t>(varName);
                PyObject* pyint = Py_BuildValue("i", tempVal);
                // I believe this will return NULL if the string is invalid
                PyObject* enumstr = Py_BuildValue("s", tempName);
                if (enumstr) {
                    PyDict_SetItem(dict,pyint,enumstr);
                    Py_DECREF(enumstr); // transfer ownership to parent
                }
            }
        }
    }

    return parent;
}

static void dictAssign(PyDgramObject* pyDgram, DescData& descdata, Xtc* myXtc)
{
    Names& names = descdata.nameindex().names();


    char keyName[2*TMPSTRINGSIZE];
    char tempName[TMPSTRINGSIZE];
    for (unsigned i = 0; i < names.num(); i++) {
        Name& name = names.get(i);
        const char* varName = name.name();
        PyObject* newobj=0; // some types don't get added here (e.g. enumdict)

        if (name.rank() == 0 || name.type()==Name::CHARSTR) {
            switch (name.type()) {
            case Name::UINT8: {
                const auto tempVal = descdata.get_value<uint8_t>(varName);
                newobj = Py_BuildValue("B", tempVal);
                break;
            }
            case Name::UINT16: {
                const auto tempVal = descdata.get_value<uint16_t>(varName);
                newobj = Py_BuildValue("H", tempVal);
                break;
            }
            case Name::UINT32: {
                const auto tempVal = descdata.get_value<uint32_t>(varName);
                newobj = Py_BuildValue("I", tempVal);
                break;
            }
            case Name::UINT64: {
                const auto tempVal = descdata.get_value<uint64_t>(varName);
                newobj = Py_BuildValue("K", tempVal);
                break;
            }
            case Name::INT8: {
                const auto tempVal = descdata.get_value<int8_t>(varName);
                newobj = Py_BuildValue("b", tempVal);
                break;
            }
            case Name::INT16: {
                const auto tempVal = descdata.get_value<int16_t>(varName);
                newobj = Py_BuildValue("h", tempVal);
                break;
            }
            case Name::INT32: {
                const auto tempVal = descdata.get_value<int32_t>(varName);
                // cpo: thought that "l" (long int) would work here
                // as well, but empirically it doesn't.
                newobj = Py_BuildValue("i", tempVal);
                break;
            }
            case Name::INT64: {
                const auto tempVal = descdata.get_value<int64_t>(varName);
                newobj = Py_BuildValue("L", tempVal);
                break;
            }
            case Name::FLOAT: {
                const auto tempVal = descdata.get_value<float>(varName);
                newobj = Py_BuildValue("f", tempVal);
                break;
            }
            case Name::DOUBLE: {
                const auto tempVal = descdata.get_value<double>(varName);
                newobj = Py_BuildValue("d", tempVal);
                break;
            }
            case Name::CHARSTR: {
                if (name.rank()!=1)
                    throw std::runtime_error("dgram.cc: string with rank != 1");
                auto arr = descdata.get_array<char>(i);
                uint32_t* shape = descdata.shape(name);
                if (strlen(arr.data())>shape[0])
                    throw std::runtime_error("dgram.cc: unterminated string");
                newobj = Py_BuildValue("s", arr.data());
                break;
            }
            case Name::ENUMVAL: {
                newobj = createEnum(varName, pyDgram, descdata);

                // overwrite the delimiter with the null character
                // so the value's python name doesn't include the dict
                // name (which follows the EnumDelim).
                strncpy(tempName,varName,TMPSTRINGSIZE-1);
                char* delim = strchr(tempName,EnumDelim);
                if (!delim) throw std::runtime_error("dgram.cc: failed to find delimitor in enum");

                *delim = '\0';
                // tell the object adder to use our modified name
                varName = tempName;
                break;
            }
            default: {
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
                auto arr = descdata.get_array<uint8_t>(i);
                newobj = PyArray_SimpleNewFromData(name.rank(), dims,
                                                   NPY_UINT8, arr.data());
                break;
            }
            case Name::UINT16: {
                auto arr = descdata.get_array<uint16_t>(i);
                newobj = PyArray_SimpleNewFromData(name.rank(), dims,
                                                   NPY_UINT16, arr.data());
                break;
            }
            case Name::UINT32: {
                auto arr = descdata.get_array<uint32_t>(i);
                newobj = PyArray_SimpleNewFromData(name.rank(), dims,
                                                   NPY_UINT32, arr.data());
                break;
            }
            case Name::UINT64: {
                auto arr = descdata.get_array<uint64_t>(i);
                newobj = PyArray_SimpleNewFromData(name.rank(), dims,
                                                   NPY_UINT64, arr.data());
                break;
            }
            case Name::INT8: {
                auto arr = descdata.get_array<int8_t>(i);
                newobj = PyArray_SimpleNewFromData(name.rank(), dims,
                                                   NPY_INT8, arr.data());
                break;
            }
            case Name::INT16: {
                auto arr = descdata.get_array<int16_t>(i);
                newobj = PyArray_SimpleNewFromData(name.rank(), dims,
                                                   NPY_INT16, arr.data());
                break;
            }
            case Name::INT32: {
                auto arr = descdata.get_array<int32_t>(i);
                newobj = PyArray_SimpleNewFromData(name.rank(), dims,
                                                   NPY_INT32, arr.data());
                break;
            }
            case Name::INT64: {
                auto arr = descdata.get_array<int64_t>(i);
                newobj = PyArray_SimpleNewFromData(name.rank(), dims,
                                                   NPY_INT64, arr.data());
                break;
            }
            case Name::FLOAT: {
                auto arr = descdata.get_array<float>(i);
                newobj = PyArray_SimpleNewFromData(name.rank(), dims,
                                                   NPY_FLOAT, arr.data());
                break;
            }
            case Name::DOUBLE: {
                auto arr = descdata.get_array<double>(i);
                newobj = PyArray_SimpleNewFromData(name.rank(), dims,
                                                   NPY_DOUBLE, arr.data());
                break;
            }
            default: {
                throw std::runtime_error("dgram.cc: Unsupported array type");
                break;
            }
            }
            if (PyArray_SetBaseObject((PyArrayObject*)newobj, pyDgram->dgrambytes) < 0) {
                printf("Failed to set BaseObject for numpy array.\n");
            }
            // PyArray_SetBaseObject steals a reference to the dgrambytes
            // but we want the dgram to also keep a reference to it as well.
            Py_INCREF(pyDgram->dgrambytes);

            // make the raw data arrays read-only
            PyArray_CLEARFLAGS((PyArrayObject*)newobj, NPY_ARRAY_WRITEABLE);
        }
        if (newobj) {
            snprintf(keyName,sizeof(keyName),"%s%s%s%s%s",
                     names.detName(),PyNameDelim,names.alg().name(),
                     PyNameDelim,varName);
            addObjHierarchy((PyObject*)pyDgram, pyDgram->contInfo.pycontainertype, keyName, newobj, names.segment());
            setXtcForSegment((PyObject*)pyDgram, pyDgram->contInfo.pycontainertype, names.detName(), names.segment(), myXtc);
        }
    }
}

class PyConvertIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    PyConvertIter(Xtc* xtc, const void* bufEnd, PyDgramObject* pyDgram, NamesLookup& namesLookup) :
        XtcIterator(xtc, bufEnd), _pyDgram(pyDgram), _namesLookup(namesLookup)
    {
    }

    int process(Xtc* xtc, const void* bufEnd)
    {
        switch (xtc->contains.id()) { //enum Type { Parent, ShapesData, Shapes, Data, Names, NumberOf };
        case (TypeId::Parent): {
            iterate(xtc, bufEnd); // look inside anything that is a Parent
            break;
        }
        case (TypeId::ShapesData): {
            ShapesData& shapesdata = *(ShapesData*)xtc;
            // lookup the index of the names we are supposed to use
            NamesId namesId = shapesdata.namesId();
            // protect against the fact that this namesid
            // may not have a NamesLookup.  cpo thinks this
            // should be fatal, since it is a sign the xtc is "corrupted",
            // in some sense.
            if (_namesLookup.count(namesId)>0) {
                DescData descdata(shapesdata, _namesLookup[namesId]);
                dictAssign(_pyDgram, descdata, xtc);
            } else {
                printf("*** Corrupt xtc: namesid 0x%x not found in NamesLookup\n",(int)namesId);
                throw "invalid namesid";
            }
            break;
        }
        default:
            break;
        }
        return Continue;
    }

private:
    PyDgramObject* _pyDgram;
    NamesLookup&      _namesLookup;
};

static void assignDict(PyDgramObject* self, PyDgramObject* configDgram) {
    bool isConfig;
    isConfig = (configDgram == 0) ? true : false;

    if (isConfig) {
        configDgram = self; // we weren't passed a config, so we must be config

        auto configSize = sizeof(Dgram) + configDgram->dgram->xtc.sizeofPayload();
        const void* configEnd = (char*)configDgram->dgram + configSize;
        configDgram->namesIter = new NamesIter(&(configDgram->dgram->xtc), configEnd);
        configDgram->namesIter->iterate();

        dictAssignConfig(configDgram, configDgram->namesIter->namesLookup());
    } else {
        self->namesIter = 0; // in case dgram was not created via dgram_init
    }

    auto size = sizeof(Dgram) + self->dgram->xtc.sizeofPayload();
    const void* bufEnd = (char*)(self->dgram) + size;
    PyConvertIter iter(&self->dgram->xtc, bufEnd, self, configDgram->namesIter->namesLookup());
    iter.iterate();
}

static void dgram_dealloc(PyDgramObject* self)
{
    Py_XDECREF(self->dict);
    if (self->namesIter) delete self->namesIter; // for config dgram only
    if (self->buf.buf == NULL) {
        // can be NULL if we had a problem early in dgram_init
        //Py_XDECREF(self->dgrambytes);

        // mona: still not sure why this prevents crashing.
        // https://docs.python.org/2/c-api/refcounting.html says that the difference is
        // the object is set to NULL prior to decrementing its reference count.
        Py_CLEAR(self->dgrambytes);
    } else {
        PyBuffer_Release(&(self->buf));
    }
    // can be NULL if we had a problem early in dgram_init
    Py_XDECREF(self->contInfo.containermod);
    Py_XDECREF(self->contInfo.pycontainertype);

    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* dgram_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    PyDgramObject* self;
    self = (PyDgramObject*)type->tp_alloc(type, 0);
    if (self != NULL) self->dict = PyDict_New();
    return (PyObject*)self;
}

ssize_t read_with_retries(int fd, void* buf, ssize_t count, size_t offset, int max_retries)
{
    ssize_t readSuccess = 0;
    for (int i=0; i<max_retries+1; i++) {
        if (i>0) {
            cout << "dgram read retry#" << i << " (max_retries=" << max_retries << ")" << endl;
        }
        if (offset == 0) {
            readSuccess = read(fd, buf, count);
        } else {
            readSuccess = pread(fd, buf, count, offset);
        }
        // see if we read all the bytes we wanted
        if (readSuccess != count) readSuccess=0;

        // sleep if reads return 0 - mona: add way to find out if end
        // of file is reached.
        if (readSuccess == 0) {
            sleep(SLEEP_SECS);
        } else {
            break;
        }
    }

    return readSuccess;
}
static int dgram_read(PyDgramObject* self, int sequential)
{
    ssize_t readSuccess=0;
    if (sequential) {
        // When read sequentially, self->dgram already has header data
        // - only reads the payload content if it is larger than 0.
        ssize_t sizeofPayload = self->dgram->xtc.sizeofPayload();
        if (sizeofPayload>0) {
            readSuccess = read_with_retries(self->file_descriptor, self->dgram->xtc.payload(), sizeofPayload, 0, self->max_retries);
        } else {
            readSuccess = 1; //no payload
        }

    } else {
        off_t fOffset = (off_t)self->offset;
        readSuccess = read_with_retries(self->file_descriptor, self->dgram, self->size, fOffset, self->max_retries);
    }

    //cout << "dgram_read offset=" << self->offset << " size=" << self->size << " readSuccess=" << readSuccess << endl;
    return readSuccess;
}

static PyObject* service(PyDgramObject* self) {
    return PyLong_FromLong(self->dgram->service());
}

static PyObject* timestamp(PyDgramObject* self) {
  return PyLong_FromLong(self->dgram->time.value());
}

Dgram& createTransition(TransitionId::Value transId, unsigned sec, unsigned usec) {
    TypeId tid(TypeId::Parent, 0);
    uint32_t env = 0;
    void* buf = malloc(sizeof(Dgram));
    Transition tr(Dgram::Event, transId, TimeStamp(sec, usec), env);
    return *new(buf) Dgram(tr, Xtc(tid));
}

static int dgram_init(PyDgramObject* self, PyObject* args, PyObject* kwds)
{
    static char* kwlist[] = {(char*)"file_descriptor",
                             (char*)"config",
                             (char*)"offset",
                             (char*)"size",
                             (char*)"view",
                             (char*)"fake_endrun",
                             (char*)"fake_endrun_sec",
                             (char*)"fake_endrun_usec",
                             (char*)"max_retries",
                             NULL};

    self->namesIter = 0;
    int fd=-1;
    PyObject* configDgram=0;
    self->offset=0;
    self->size=0;
    bool isView=0;
    PyObject* view=0;
    int fake_endrun=0;
    unsigned fake_endrun_sec=0;
    unsigned fake_endrun_usec=0;
    self->max_retries=0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
                                     "|iOllOiIIi", kwlist,
                                     &fd,
                                     &configDgram,
                                     &self->offset,
                                     &self->size,
                                     &view,
                                     &fake_endrun,
                                     &fake_endrun_sec,
                                     &fake_endrun_usec,
                                     &self->max_retries)) {
        return -1;
    }

    if (fd > -1) {
        if (fcntl(fd, F_GETFD) == -1) {
            PyErr_SetString(PyExc_OSError, "invalid file descriptor");
            return -1;
        }
    }

    isView = (view!=0) ? true : false;

    self->contInfo.containermod = PyImport_ImportModule("psana.container");
    self->contInfo.pycontainertype = PyObject_GetAttrString(self->contInfo.containermod,"Container");

    // Retrieve size and file_descriptor
    Dgram dgram_header; // For case (1) and (2) below to store the header for later
    if (!isView) {
        // If view is not given, we assume dgram is to be created as one of these:
        // 1. Dgram(file_descriptor=fd) --> create config dgram by read
        // 2. Dgram(config=config)          create data dgram by read
        // 3. Dgram(file_descriptor=fd, config=config, offset=int, size=int)
        //                                  create data dgram by pread
        // 4. Dgram(config=config, fake_endrun=1, fake_endrun_sec=0, fake_endrun_usec=0)

        if (fd==-1 && configDgram==0) {
            PyErr_SetString(PyExc_RuntimeError, "Creating empty dgram is no longer supported.");
            return -1;
        } else {
            if (fd==-1) {
                // For (2) and (4)
                self->file_descriptor=((PyDgramObject*)configDgram)->file_descriptor;
            } else {
                // For (1) and (3)
                self->file_descriptor=fd;
            }

            // We know size for 3 (already set at parsed arg) and 4
            if (fake_endrun == 1) {
                self->size = sizeof(Dgram);
            }

            // For (1) and (2),
            if (self->size == 0) {
                // For (1) and (2), obtain dgram_header from fd then extract size
                int readSuccess = read_with_retries(self->file_descriptor, &dgram_header, sizeof(Dgram), 0, self->max_retries);
                if (readSuccess <= 0) {
                    PyErr_SetString(PyExc_StopIteration, "Problem reading dgram header.");
                    return -1;
                }

                self->size = sizeof(Dgram) + dgram_header.xtc.sizeofPayload();
            }
        }

        if (self->size == 0) {
            PyErr_SetString(PyExc_RuntimeError, "Can't retrieve dgram size. Either size is not given when creating read-by-offset dgram or there's a problem reading dgram header.");
            return -1;
        }

        //PyObject* arglist = Py_BuildValue("(i)",self->size);
        // I believe this memsets the buffer to 0, which we don't need.
        // Perhaps ideally we would write a custom object to avoid this. - cpo
        //self->dgrambytes = PyObject_CallObject((PyObject*)&PyByteArray_Type, arglist);
        //Py_DECREF(arglist);

        // Use c-level api to create PyByteArray to avoid memset - mona
        self->dgrambytes = PyByteArray_FromStringAndSize(NULL, self->size);
        if (self->dgrambytes == NULL) {
            return -1;
        }

        if (fake_endrun == 1) {
            Dgram& endRunTr = createTransition(TransitionId::EndRun, fake_endrun_sec, fake_endrun_usec);
            self->dgram = &endRunTr;
        } else {
            self->dgram = (Dgram*)(PyByteArray_AS_STRING(self->dgrambytes));
        }

    } else { // if (!isview) {
        // Creating a dgram from view (any objects with a buffer interface) can be done by:
        // 5. Dgram(view=view, offset=int) --> create a config dgram from the view
        // 6. Dgram(view=view, config=config, offset=int) --> create a data dgram using the config and view

        // this next line is needed because arrays will increase the reference count
        // of the view (actually a PyByteArray) in dictAssign.  This is the mechanism we
        // use so we don't have to copy the array data.
        self->dgrambytes = view;
        if (PyObject_GetBuffer(view, &(self->buf), PyBUF_SIMPLE) == -1) {
            PyErr_SetString(PyExc_MemoryError, "unable to create dgram with the given view");
            return -1;
        }
        self->dgram = (Dgram*)(((char *)self->buf.buf) + self->offset);
        self->size = sizeof(Dgram) + self->dgram->xtc.sizeofPayload();
    } // else if (!isView)

    if (self->dgram == NULL) {
        PyErr_SetString(PyExc_MemoryError, "insufficient memory to create Dgram object");
        return -1;
    }

    // Read the data if this dgram is not a view
    if (!isView && fake_endrun == 0) {
        bool sequential = (fd==-1) != (configDgram==0);
        if (sequential) {
          memcpy((void*)self->dgram, (const void*)&dgram_header, sizeof(dgram_header));
        }

        ssize_t readSuccess = dgram_read(self, sequential);
        if (readSuccess == 0) {
            char s[TMPSTRINGSIZE];
            printf("dgram.cc: , dgram read error raising StopIteration.\n");
            snprintf(s, sizeof(s), "loading dgram was unsuccessful -- %s", strerror(errno));
            PyErr_SetString(PyExc_StopIteration, s);
            return -1;
        }
    }

    // In case we got a renew config (second or more config dgram), we have to
    // clear the given config (configDgram) to allow the next routines to use
    // self as a config and assign dictionary to it.
    if (self->dgram->service() == TransitionId::Configure) {
        configDgram = 0;
    }

    assignDict(self, (PyDgramObject*)configDgram);

    // Add top level xtc container and its attributes
    setXtc((PyObject*)self, self->contInfo.pycontainertype, &(self->dgram->xtc));

    return 0;
}

#if PY_MAJOR_VERSION < 3
static Py_ssize_t PyDgramObject_getsegcount(PyDgramObject *self, Py_ssize_t *lenp) {
    return 1; // only supports single segment
}

static Py_ssize_t PyDgramObject_getreadbuf(PyDgramObject *self, Py_ssize_t segment, void **ptrptr) {
    *ptrptr = (void*)self->dgram;
    return self->size;
}

static Py_ssize_t PyDgramObject_getwritebuf(PyDgramObject *self, Py_ssize_t segment, void **ptrptr) {
    return PyDgramObject_getreadbuf(self, segment, (void **)ptrptr);
}

static Py_ssize_t PyDgramObject_getcharbuf(PyDgramObject *self, Py_ssize_t segment, constchar **ptrptr) {
    return PyDgramObject_getreadbuf(self, segment, (void **) ptrptr);
}
#endif /* PY_MAJOR_VERSION < 3 */

static int PyDgramObject_getbuffer(PyObject *obj, Py_buffer *view, int flags)
{
    if (view == 0) {
        PyErr_SetString(PyExc_ValueError, "NULL view in getbuffer");
        return -1;
    }

    PyDgramObject* self = (PyDgramObject*)obj;
    view->obj = (PyObject*)self;
    view->buf = (void*)self->dgram;
    view->len = self->size;
    view->readonly = 1;
    view->itemsize = 1;
    view->format = (char *)"s";
    view->ndim = 1;
    view->shape = &view->len;
    view->strides = &view->itemsize;
    view->suboffsets = NULL;
    view->internal = NULL;

    Py_INCREF(self);
    return 0;
}

static PyBufferProcs PyDgramObject_as_buffer = {
#if PY_MAJOR_VERSION < 3
    (readbufferproc)PyDgramObject_getreadbuf,   /*bf_getreadbuffer*/
    (writebufferproc)PyDgramObject_getwritebuf, /*bf_getwritebuffer*/
    (segcountproc)PyDgramObject_getsegcount,    /*bf_getsegcount*/
    (charbufferproc)PyDgramObject_getcharbuf,   /*bf_getcharbuffer*/
#endif
    // this definition is only compatible with Python 3.3 and above
    (getbufferproc)PyDgramObject_getbuffer,
    (releasebufferproc)0, // no special release required
};

static PyMemberDef dgram_members[] = {
    { (char*)"__dict__",
      T_OBJECT_EX, offsetof(PyDgramObject, dict),
      0,
      (char*)"attribute dictionary" },
    { (char*)"_file_descriptor",
      T_INT, offsetof(PyDgramObject, file_descriptor),
      0,
      (char*)"attribute file_descriptor" },
    { (char*)"_offset",
      T_INT, offsetof(PyDgramObject, offset),
      0,
      (char*)"attribute offset" },
    { (char*)"_size",
      T_INT, offsetof(PyDgramObject, size),
      0,
      (char*)"size (bytes) of the dgram" },
    { (char*)"_dgrambytes",
      T_OBJECT_EX, offsetof(PyDgramObject, dgrambytes),
      0,
      (char*)"attribute offset" },
    { NULL }
};

static PyMethodDef dgram_methods[] = {
    {"service", (PyCFunction)service, METH_NOARGS, "service"},
    {"timestamp", (PyCFunction)timestamp, METH_NOARGS, "timestamp"},
    {NULL}  /* Sentinel */
};

static PyTypeObject dgram_DgramType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "psana.dgram.Dgram", /* tp_name */
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
    0, /* tp_getattro */
    0, /* tp_setattro */
    &PyDgramObject_as_buffer, /* tp_as_buffer */
    (Py_TPFLAGS_DEFAULT
#if PY_MAJOR_VERSION < 3
    | Py_TPFLAGS_CHECKTYPES
    | Py_TPFLAGS_HAVE_NEWBUFFER
#endif
    | Py_TPFLAGS_BASETYPE),   /* tp_flags */
    0, /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    0, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    0, /* tp_iter */
    0, /* tp_iternext */
    dgram_methods, /* tp_methods */
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

#if PY_MAJOR_VERSION > 2
static PyModuleDef dgrammodule =
{ PyModuleDef_HEAD_INIT, "dgram", NULL, -1, NULL, NULL, NULL, NULL, NULL };
#endif

#ifndef PyMODINIT_FUNC /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

#if PY_MAJOR_VERSION > 2
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
#else
PyMODINIT_FUNC initdgram(void) {
    PyObject *m;

    import_array();

    if (PyType_Ready(&dgram_DgramType) < 0)
        return;

    m = Py_InitModule3("dgram", dgram_methods, "Dgram module.");

    if (m == NULL)
        return;

    Py_INCREF(&dgram_DgramType);
    PyModule_AddObject(m, "Dgram", (PyObject *)&dgram_DgramType);
}
#endif
