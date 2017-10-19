#ifndef SHAPESDATA__H
#define SHAPESDATA__H

#include <vector>
#include <cstring>
#include <assert.h>

#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/TypeId.hh"

class Name
{
public:
    enum DataType { UINT8, UINT16, INT32, FLOAT, DOUBLE };

    static int get_element_size(DataType type);

    enum {MaxRank=5};
    static const int maxNameSize = 256;
    Name(const char* name, DataType type, int rank)
    {
        strncpy(_name, name, maxNameSize);
        _type = type;
        _rank = rank;
    }
    const char* name() {return _name;}
    DataType    type() {return _type;}
    uint32_t    rank() {return _rank;}

private:
    char _name[maxNameSize];
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
    Names() : AutoParentAlloc(XtcData::TypeId(XtcData::TypeId::Names,0)) {}
    Name& get(unsigned index)
    {
        Name& name = ((Name*)(this + 1))[index];
        return name;
    }

    unsigned num()
    {
        return sizeofPayload() / sizeof(Name);
    }
    // Add new item to Names
    void add(const char* name, Name::DataType type, XtcData::Xtc& parent, int rank=0)
    {
        void* ptr = alloc(sizeof(Name), parent);
        new (ptr) Name(name, type, rank);
    }
};

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
        // go two levels up to "auto-alloc" Shapes Xtc header size
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
            assert(d.contains.id()==XtcData::TypeId::Shapes);
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
