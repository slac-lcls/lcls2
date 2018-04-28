#ifndef SHAPESDATA__H
#define SHAPESDATA__H

#include <vector>
#include <cstring>
#include <assert.h>
#include <iostream>

#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/TypeId.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/Dgram.hh"
namespace XtcData {
class VarDef;


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

    uint32_t version() {
        return _version.version();
    }

    const char* name() {return _alg;}

private:
    char _alg[maxNameSize];
    AlgVersion _version;
};

class Name {
public:
  enum DataType { UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64, FLOAT, DOUBLE};

    static int get_element_size(DataType type);

    enum {MaxRank=5};
  
    Name(const char* name, DataType type, int rank=0) : _alg("",0,0,0) {
        // For compatibility with python
        //assert(rank != 0 && (type=INT64 || type = DOUBLE));

      // Assert maxrank, maxnamesize
      assert(rank < MaxRank);assert(strlen(name) < maxNameSize);
      strncpy(_name, name, maxNameSize);
        _type = type;
        _rank = rank;
    }
  
    Name(const char* name, DataType type, int rank, Alg& alg) : _alg(alg) {
        assert(rank < MaxRank);assert(sizeof(name) < maxNameSize);      
        strncpy(_name, name, maxNameSize);
        _type = type;
        _rank = rank;
    } 

    Name(const char* name, Alg& alg) : _alg(alg) {
	assert(sizeof(name) < maxNameSize);
        strncpy(_name, name, maxNameSize);
        _type = Name::UINT8;
        _rank = 1;
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
  Shape(uint32_t shape[Name::MaxRank])
    {
        memcpy(_shape, shape, sizeof(uint32_t) * Name::MaxRank);
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
class AutoParentAlloc : public Xtc
{
public:
    AutoParentAlloc(TypeId typeId) : Xtc(typeId)
    {
    }
    void* alloc(uint32_t size, Xtc& parent) {
        parent.alloc(size);
        return Xtc::alloc(size);
    }
    void* alloc(uint32_t size, Xtc& parent, Xtc& superparent) {
        superparent.alloc(size);
        parent.alloc(size);
        return Xtc::alloc(size);
    }
};

// in principal this should be an arbitrary hierarchy of xtc's.
// e.g. detName.detAlg.subfield1.subfield2...
// but for code simplicity keep it to one Names xtc, which holds
// both detName/detAlg, and all the subfields are encoded in the Name
// objects using a delimiter, currently "_".
// Having an arbitrary xtc hierarchy would
// create complications in maintaining all the xtc extents.
// perhaps should split this class into two xtc's: the Alg part (DataNames?)
// and the detName/detType/segment part (DetInfo?).  but then
// if there are multiple detectors in an xtc need to come up with another
 /// mechanism for the DataName to point to the correct DetInfo.


class NameInfo
{
public:
    uint32_t numArrays;
    char     detType[maxNameSize];
    char     detName[maxNameSize];
    char     detId[maxNameSize];
    Alg      alg;
    uint32_t segment;

    NameInfo(const char* detname, Alg& alg0, const char* dettype, const char* detid, uint32_t segment0, uint32_t numarr=0):alg(alg0), segment(segment0){
        numArrays = numarr;
        strncpy(detName, detname, maxNameSize);
        strncpy(detType, dettype, maxNameSize);
        strncpy(detId,   detid,   maxNameSize);
    }


};


class Names : public AutoParentAlloc
{
public:


    Names(const char* detName, Alg& alg, const char* detType, const char* detId, unsigned segment=0) :
        AutoParentAlloc(TypeId(TypeId::Names,0)),
        _NameInfo(detName, alg, detType, detId, segment)
    {

        // allocate space for our private data
        Xtc::alloc(sizeof(*this)-sizeof(AutoParentAlloc));
    }

    uint32_t numArrays(){return _NameInfo.numArrays;}; 
    const char* detName() {return _NameInfo.detName;}
    const char* detType() {return _NameInfo.detType;}
    const char* detId()   {return _NameInfo.detId;}
    unsigned    segment() {return _NameInfo.segment;}
    Alg&        alg()     {return _NameInfo.alg;}

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


    void add(Xtc& parent, VarDef& V)
    {
      for(auto const & elem: V.NameVec)
      	{
      	  void* ptr = alloc(sizeof(Name), parent);
      	  new (ptr) Name(elem);

	  if(Name(elem).rank() > 0){_NameInfo.numArrays++;};
      	};

    }
private:
    NameInfo _NameInfo;
};

#include <stdio.h>
class Data : public AutoParentAlloc
{
public:
    Data(Xtc& superparent) : 
        AutoParentAlloc(TypeId(TypeId::Data,0))
    {
        // go two levels up to "auto-alloc" Data Xtc header size
        superparent.alloc(sizeof(*this));
    }
};

class Shapes : public AutoParentAlloc
{
public:
    Shapes(Xtc& superparent, uint32_t namesId) :
        AutoParentAlloc(TypeId(TypeId::Shapes,0)),
        _namesId(namesId)
    {
        // allocate space for our private data
        Xtc::alloc(sizeof(*this)-sizeof(AutoParentAlloc));
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

class ShapesData : public Xtc
{
public:
    ShapesData() : Xtc(TypeId(TypeId::ShapesData,0)) {}

    Data& data()
    {
        if (_firstIsShapes()) {
            Data& d = reinterpret_cast<Data&>(_second());
            assert(d.contains.id()==TypeId::Data);
            return d;
        }
        else {
            Data& d = reinterpret_cast<Data&>(_first());
            assert(d.contains.id()==TypeId::Data);
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
            assert(d.contains.id()==TypeId::Shapes);
            return d;
        }
    }

private:
    Xtc& _first() {
        return *(Xtc*)payload();
    }

    Xtc& _second() {
        return *_first().next();
    }

    bool _firstIsShapes() {
        return _first().contains.id()==TypeId::Shapes;
    }


};

class blockDgram : public Xtc
{
public:
    blockDgram(uint8_t* buffdgram):_dgram(*(Dgram*)buffdgram){
        // _Dgram& dgram = *(Dgram*)buffdgram;
        TypeId tid(TypeId::Parent, 0);
        _dgram.xtc.contains = tid;
        _dgram.xtc.damage = 0;
        _dgram.xtc.extent = sizeof(Xtc);

        _sizeDgram =sizeof(Dgram)+_dgram.xtc.sizeofPayload();
    };

    void addNamesBlock(uint8_t* name_block, size_t block_elems){
        Xtc& namesxtc = *new((char*)_dgram.xtc.alloc(sizeof(Xtc))) Xtc(TypeId(TypeId::Names, 0));
        size_t nameblock_size = sizeof(NameInfo) + block_elems*sizeof(Name);
        memcpy(namesxtc.payload(), name_block, nameblock_size);
        namesxtc.alloc(nameblock_size);
        _dgram.xtc.alloc(nameblock_size);

        _sizeDgram =sizeof(Dgram)+_dgram.xtc.sizeofPayload();
    }

    void addShapesDataBlock(uint8_t* shape_block, uint8_t* data_block, size_t sizeofdata, size_t block_elems){
        Xtc& shapesdata = *new((char*)_dgram.xtc.alloc(sizeof(Xtc))) Xtc(TypeId(TypeId::ShapesData, 0));

        Xtc& shapes = *new((char*)shapesdata.alloc(sizeof(Xtc))) Xtc(TypeId(TypeId::Shapes, 0));
        size_t shapeblock_size = sizeof(uint32_t) + block_elems*sizeof(Shape);
        memcpy(shapes.payload(), shape_block, shapeblock_size);
        shapes.alloc(shapeblock_size);
        shapesdata.alloc(shapeblock_size);
        _dgram.xtc.alloc(shapeblock_size+sizeof(Xtc));


        Xtc& data = *new((char*)shapesdata.alloc(sizeof(Xtc))) Xtc(TypeId(TypeId::Data, 0));
        memcpy(data.payload(), data_block, sizeofdata);

        data.alloc(sizeofdata);
        shapesdata.alloc(sizeofdata);
        _dgram.xtc.alloc(sizeofdata+sizeof(Xtc));
        _sizeDgram =sizeof(Dgram)+_dgram.xtc.sizeofPayload();
    }

    void addDataBlock(uint8_t* data_block, size_t sizeofdata){
        Xtc& shapesdata = *new((char*)_dgram.xtc.alloc(sizeof(Xtc))) Xtc(TypeId(TypeId::ShapesData, 0));

        Xtc& data = *new((char*)shapesdata.alloc(sizeof(Xtc))) Xtc(TypeId(TypeId::Data, 0));
        memcpy(data.payload(), data_block, sizeofdata);

        data.alloc(sizeofdata);
        shapesdata.alloc(sizeofdata);
        _dgram.xtc.alloc(sizeofdata+sizeof(Xtc));
        _sizeDgram =sizeof(Dgram)+_dgram.xtc.sizeofPayload();
    }

    uint32_t dgramSize(){
        return _sizeDgram;
    };

private:
    size_t _sizeDgram = 0;
    Dgram& _dgram;

};
};


#endif // SHAPESDATA__H
