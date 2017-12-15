#ifndef SHAPESDATA__H
#define SHAPESDATA__H

#include <vector>
#include <cstring>
#include <assert.h>

#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/TypeId.hh"

static const int maxNameSize = 256;

class AlgVersion {
public:
    AlgVersion(uint8_t major, uint8_t minor, uint8_t micro) {
        _version = major<<16 | minor<<8 | micro;
    }
    unsigned major() {return (_version>>16)&0xff;}
    unsigned minor() {return (_version>>8)&0xff;}
    unsigned micro() {return (_version)&0xff;}
    unsigned version() {return _version;}
private:
    uint32_t _version;
};

class Alg {
public:
    Alg(const char* alg, uint8_t major, uint8_t minor, uint8_t micro) :
        _version(major,minor,micro) {
        strncpy(_alg, alg, maxNameSize);
    }

    Alg(Alg& other) : _version(other._version) {
        strncpy(_alg, other._alg, maxNameSize);
    }

    uint32_t getVersion() {
        return _version.version();
    }

    const char* getAlgName() {return _alg;}

private:
    char _alg[maxNameSize];
    AlgVersion _version;
};

class Name {
public:
    enum DataType { UINT8, UINT16, INT32, FLOAT, DOUBLE };

    static int get_element_size(DataType type);

    enum {MaxRank=5};
    Name(Alg& alg, const char* name, DataType type, int rank) : _alg(alg) {
        strncpy(_name, name, maxNameSize);
        _type = type;
        _rank = rank;
    }

    const char* name() {return _name;}
    DataType    type() {return _type;}
    uint32_t    rank() {return _rank;}
    Alg&        alg()  {return _alg;}

private:
    Alg      _alg;
    char     _name[maxNameSize];
    DataType _type;
    uint32_t _rank;
};

class Shape
{
public:
    Shape(unsigned shape[Name::MaxRank])
    {
        memcpy(_shape, shape, sizeof(int) * Name::MaxRank);
    }
    unsigned size(Name& name) {
        unsigned size = 1;
        for (unsigned i = 0; i < name.rank(); i++) {
            size *= _shape[i];
        }
        return size*Name::get_element_size(name.type());
    }
    uint32_t* shape() {return _shape;}
private:
    uint32_t _shape[Name::MaxRank]; // in an ideal world this would have variable length "rank"
};

// this class updates the "parent" Xtc extent at the same time
// the child Shapes or Data extent is increased.  the hope is that this
// will make management of the Xtc's less error prone, at the price
// of some performance (more calls to Xtc::alloc())
class AutoParentAlloc : public XtcData::Xtc
{
public:
    AutoParentAlloc(XtcData::TypeId typeId) : XtcData::Xtc(typeId)
    {
    }
    void* alloc(uint32_t size, XtcData::Xtc& parent) {
        parent.alloc(size);
        return XtcData::Xtc::alloc(size);
    }
    void* alloc(uint32_t size, XtcData::Xtc& parent, XtcData::Xtc& superparent) {
        superparent.alloc(size);
        parent.alloc(size);
        return XtcData::Xtc::alloc(size);
    }
};

class Names : public AutoParentAlloc
{
public:

    Names(const char* detName, const char* dataName ) :
        AutoParentAlloc(XtcData::TypeId(XtcData::TypeId::Names,0))
    {
        strncpy(_dataName, dataName, maxNameSize);
        strncpy(_detName, detName, maxNameSize);
        // allocate space for our private data
        XtcData::Xtc::alloc(sizeof(*this)-sizeof(AutoParentAlloc));
    }

    const char* dataName() {return _dataName;}
    const char* detName()  {return _detName;}

    Name& get(unsigned index)
    {
        Name& name = ((Name*)(this + 1))[index];
        return name;
    }

    unsigned num()
    {
        unsigned sizeOfNames = (char*)next()-(char*)(this+1);
        assert (sizeOfNames%sizeof(Name)==0);
        return sizeOfNames / sizeof(Name);
    }

    // Add new item to Names
    void add(Alg& alg, const char* name, Name::DataType type, XtcData::Xtc& parent, int rank=0)
    {
        void* ptr = alloc(sizeof(Name), parent);
        new (ptr) Name(alg, name, type, rank);
    }
private:
    char     _dataName[maxNameSize];
    char     _detName[maxNameSize];
};

#include <stdio.h>
class Data : public AutoParentAlloc
{
public:
    Data(XtcData::Xtc& superparent) : 
        AutoParentAlloc(XtcData::TypeId(XtcData::TypeId::Data,0))
    {
        // go two levels up to "auto-alloc" Data Xtc header size
        superparent.alloc(sizeof(*this));
    }
};

class Shapes : public AutoParentAlloc
{
public:
    Shapes(XtcData::Xtc& superparent, uint32_t namesId) :
        AutoParentAlloc(XtcData::TypeId(XtcData::TypeId::Shapes,0)),
        _namesId(namesId)
    {
        // allocate space for our private data
        XtcData::Xtc::alloc(sizeof(*this)-sizeof(AutoParentAlloc));
        // go two levels up to "auto-alloc" Shapes size
        superparent.alloc(sizeof(*this));
    }

    Shape& get(unsigned index)
    {
        Shape& shape = ((Shape*)(this + 1))[index];
        return shape;
    }
    uint32_t namesId() {return _namesId;}
private:
    // associated numerical index of the Names object in the configure transition
    uint32_t _namesId;
};

class ShapesData : public XtcData::Xtc
{
public:
    ShapesData() : XtcData::Xtc(XtcData::TypeId(XtcData::TypeId::ShapesData,0)) {}

    Data& data()
    {
        if (_firstIsShapes()) {
            Data& d = reinterpret_cast<Data&>(_second());
            assert(d.contains.id()==XtcData::TypeId::Data);
            return d;
        }
        else {
            Data& d = reinterpret_cast<Data&>(_first());
            assert(d.contains.id()==XtcData::TypeId::Data);
            return d;
        }
    }

    Shapes& shapes()
    {
        if (_firstIsShapes()) {
            Shapes& d = reinterpret_cast<Shapes&>(_first());
            return d;
        }
        else {
            Shapes& d = reinterpret_cast<Shapes&>(_second());
            assert(d.contains.id()==XtcData::TypeId::Shapes);
            return d;
        }
    }

private:
    XtcData::Xtc& _first() {
        return *(XtcData::Xtc*)payload();
    }

    XtcData::Xtc& _second() {
        return *_first().next();
    }

    bool _firstIsShapes() {
        return _first().contains.id()==XtcData::TypeId::Shapes;
    }

};


#endif // SHAPESDATA__H
