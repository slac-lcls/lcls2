#ifndef SHAPESDATA__H
#define SHAPESDATA__H

#include <vector>
#include <cstring>
#include <iostream>
#include <stdio.h>

#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/Array.hh"
#include "xtcdata/xtc/TypeId.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/NamesId.hh"

namespace XtcData {

class VarDef;

static const int MaxNameSize = 256;

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
        strncpy(_alg, alg, MaxNameSize-1);
    }

    uint32_t version() {
        return _version.version();
    }

    const char* name() {return _alg;}

private:
    char _alg[MaxNameSize];
    AlgVersion _version;
};

class Name {
public:
    // if you add types here, you must update the corresponding sizes in ShapesData.cc
    enum DataType {UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64, FLOAT, DOUBLE, CHARSTR, ENUMVAL, ENUMDICT};

    static int get_element_size(DataType type);

    Name(const char* name, DataType type, int rank=0) : _alg("",0,0,0) {
        // We should consider using this for creating datagrams
        // from python, since python only support INT64/DOUBLE
        // (apart from arrays)
        // assert(rank != 0 && (type=INT64 || type = DOUBLE));
        if (rank >= MaxRank) {
            printf("*** %s:%d: rank %d too large for array\n",__FILE__,__LINE__,rank);
            abort();
        }
        if (strlen(name) >= MaxNameSize) {
            printf("*** %s:%d: namelength %zu too large\n",__FILE__,__LINE__,strlen(name));
            abort();
        }
        strncpy(_name, name, MaxNameSize-1);
        _type = (uint32_t)type;
        _rank = rank;
        _checkname();
    }

    Name(const char* name, DataType type, int rank, Alg& alg) : _alg(alg) {
        if (rank >= MaxRank) {
            printf("*** %s:%d: rank %d too large for array\n",__FILE__,__LINE__,rank);
            abort();
        }
        if (strlen(name) >= MaxNameSize) {
            printf("*** %s:%d: namelength %zu too large\n",__FILE__,__LINE__,strlen(name));
            abort();
        }
        strncpy(_name, name, MaxNameSize-1);
        _type = (uint32_t)type;
        _rank = rank;
        _checkname();
    }

    Name(const char* name, Alg& alg) : _alg(alg) {
        if (strlen(name) >= MaxNameSize) {
            printf("*** %s:%d: namelength %zu too large\n",__FILE__,__LINE__,strlen(name));
            abort();
        }
        strncpy(_name, name, MaxNameSize-1);
        _type = (uint32_t)Name::UINT8;
        _rank = 1;
        _checkname();
    }

    const char* name() {return _name;}
    DataType    type() {return (DataType)_type;}
    uint32_t    rank() {return _rank;}
    Alg&        alg()  {return _alg;}
    const char* str_type();


private:
    void _checkname() {
        const char* ptr = _name;
        char val;
        // check for allowed characters
        // ".": 46, "0-9": 48-57, ":": 58, "A-Z": 65-90, "_": 95, "a-z": 97-122
        // allow "." for attribute hierarchies
        // allow ":" for step-scan epics vars which have no clean python xtc name
        while(*ptr!='\0' && (ptr-_name)<MaxNameSize) {
            val=*ptr;
            if ((val<46) || (val==47) || (val>58 && val<65) || (val>90 && val<95)
                || (val>95 && val<97) || (val>122)) {
                printf("*** Error: illegal XtcData::Name: %s. Aborting.\n",_name);
                abort();
            }
            ptr++;
        }
    }

    Alg      _alg;
    char     _name[MaxNameSize];
    uint32_t _type;
    uint32_t _rank;
};



class Shape
{
public:
  Shape(uint32_t shape[MaxRank])
    {
        memcpy(_shape, shape, sizeof(uint32_t) * MaxRank);
    }
    unsigned num_elements(unsigned rank) {
        unsigned n = 1;
        for (unsigned i = 0; i < rank; i++) {
            n *= _shape[i];
        }
        return n;
    }

    unsigned size(Name& name) {
        return num_elements(name.rank())*Name::get_element_size(name.type());
    }

    uint32_t* shape() {return _shape;}
private:
    uint32_t _shape[MaxRank]; // in an ideal world this would have variable length "rank"
};

// this class updates the "parent" Xtc extent at the same time
// the child Shapes or Data extent is increased.  the hope is that this
// will make management of the Xtc's less error prone, at the price
// of some performance (more calls to Xtc::alloc())
class AutoParentAlloc : public Xtc
{
public:
    AutoParentAlloc(TypeId typeId) : Xtc(typeId) {}
    AutoParentAlloc(TypeId typeId, const NamesId& namesId) : Xtc(typeId,namesId) {}
    void* alloc(uint32_t size, Xtc& parent, const void* bufEnd) {
        parent.alloc(size, bufEnd);
        return Xtc::alloc(size, bufEnd);
    }
    void* alloc(uint32_t size, Xtc& parent, Xtc& superparent, const void* bufEnd) {
        superparent.alloc(size, bufEnd);
        parent.alloc(size, bufEnd);
        return Xtc::alloc(size, bufEnd);
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
    // This order must be preserved in order to read already recorded data
    uint32_t numArrays;
    char     detType[MaxNameSize];
    char     detName[MaxNameSize];
    char     detId[MaxNameSize];
    Alg      alg;
    uint32_t segment;

    NameInfo(const char* detname, Alg& alg0, const char* dettype, const char* detid, uint32_t segment0, uint32_t numarr=0):alg(alg0), segment(segment0){
        numArrays = numarr;
        _strncpy(detName, detname, MaxNameSize-1);
        _strncpy(detType, dettype, MaxNameSize-1);
        _strncpy(detId,   detid,   MaxNameSize-1);
    }
private:
    // Avoid GCC-8 warnings that are probably legitimate but incomprehensible
    void _strncpy(char* dst, const char* src, size_t dstLen) {
        auto srcLen = strnlen(src, dstLen);
        memcpy(dst, src, srcLen);
        dst[srcLen] = '\0';
    }
};


class Names : public AutoParentAlloc
{
public:

    Names(const void* bufEnd, const char* detName, Alg& alg, const char* detType, const char* detId, const NamesId& namesId, unsigned segment=0) :
        AutoParentAlloc(TypeId(TypeId::Names,0),namesId),
        _NameInfo(detName, alg, detType, detId, segment)
    {
        // allocate space for our private data
        Xtc::alloc(sizeof(*this)-sizeof(AutoParentAlloc), bufEnd);
        _checkname();
    }

    NamesId& namesId() {return (NamesId&)src;}

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
        if (sizeOfNames%sizeof(Name)!=0) {
            printf("*** %s:%d: Name object alignment error %u\n",__FILE__,__LINE__,unsigned(sizeOfNames%sizeof(Name)));
            abort();
        }
        return sizeOfNames / sizeof(Name);
    }


    void add(Xtc& parent, const void* bufEnd, VarDef& V)
    {
        for(auto const & elem: V.NameVec)
        {
            void* ptr = alloc(sizeof(Name), parent, bufEnd);
            new (ptr) Name(elem);

            if(Name(elem).rank() > 0){_NameInfo.numArrays++;};
        };
    }
private:
    void _checkname() {
        const char* ptr = detName();
        char val;
        // check for allowed characters
        // "0-9": 48-57, "A-Z": 65-90, "_": 95, "a-z": 97-122
        while(*ptr!='\0' && (ptr-detName())<MaxNameSize) {
            val=*ptr;
            if ((val<48) || (val>57 && val<65) || (val>90 && val<95)
                || (val>95 && val<97) || (val>122)) {
                printf("*** Error: illegal XtcData::Names detname: %s. Aborting.\n",detName());
                abort();
            }
            ptr++;
        }
    }

    NameInfo _NameInfo;
};

class Data : public AutoParentAlloc
{
public:
    Data(Xtc& superparent, const void* bufEnd) :
        AutoParentAlloc(TypeId(TypeId::Data,0))
    {
        // go two levels up to "auto-alloc" Data Xtc header size
        superparent.alloc(sizeof(*this), bufEnd);
    }
};

class Shapes : public AutoParentAlloc
{
public:
    Shapes(Xtc& superparent, const void* bufEnd) :
        AutoParentAlloc(TypeId(TypeId::Shapes,0))
    {
        // allocate space for our private data
        // not strictly necessary since we currently have no private data.
        Xtc::alloc(sizeof(*this)-sizeof(AutoParentAlloc), bufEnd);
        // go two levels up to "auto-alloc" Shapes size
        superparent.alloc(sizeof(*this), bufEnd);
    }

    Shape& get(unsigned index)
    {
        Shape& shape = ((Shape*)(this + 1))[index];
        return shape;
    }
};

class ShapesData : public Xtc
{
public:
    ShapesData(NamesId& namesId) : Xtc(TypeId(TypeId::ShapesData,0),namesId) {}

    NamesId& namesId() {return (NamesId&)src;}

    Data& data()
    {
        if (_firstIsShapes()) {
            Data& d = reinterpret_cast<Data&>(_second());
            if (d.contains.id()!=TypeId::Data) {
                printf("*** %s:%d: incorrect TypeId %d\n",__FILE__,__LINE__,d.contains.id());
                abort();
            }
            return d;
        }
        else {
            Data& d = reinterpret_cast<Data&>(_first());
            if (d.contains.id()!=TypeId::Data) {
                printf("*** %s:%d: incorrect TypeId %d\n",__FILE__,__LINE__,d.contains.id());
                abort();
            }
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
            if (d.contains.id()!=TypeId::Shapes) {
                printf("*** %s:%d: incorrect TypeId %d\n",__FILE__,__LINE__,d.contains.id());
                abort();
            }
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

}; // namespace XtcData

#endif // SHAPESDATA__H
