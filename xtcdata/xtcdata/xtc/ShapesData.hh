/**
 * @file
 * @brief Declares the self-describing data classes: AlgVersion, Alg, Name, Shape, NameInfo, the Names xtc, and the ShapesData xtc with its Shapes and Data children.
 */
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

/** Size (256) of the fixed char arrays that hold names in Alg, Name and NameInfo. */
static const int MaxNameSize = 256;

/** Version packed in 32 bits: major in bits 23:16, minor in bits 15:8, micro in bits 7:0. */
class AlgVersion {
public:
    /** Store major<<16 | minor<<8 | micro. */
    AlgVersion(uint8_t major, uint8_t minor, uint8_t micro) {
        _version = major<<16 | minor<<8 | micro;
    }
    /** Return bits 23:16. */
    unsigned major() {return (_version>>16)&0xff;}
    /** Return bits 15:8. */
    unsigned minor() {return (_version>>8)&0xff;}
    /** Return bits 7:0. */
    unsigned micro() {return (_version)&0xff;}
    /** Return the packed 32-bit value. */
    unsigned version() {return _version;}
private:
    uint32_t _version;
};

/** Algorithm name (char[MaxNameSize]) plus an AlgVersion. */
class Alg {
public:
    /**
     * Copy at most MaxNameSize-1 characters of alg with strncpy and pack the version.
     * The name is not NUL-terminated by this code if alg has MaxNameSize-1 or more characters.
     */
    Alg(const char* alg, uint8_t major, uint8_t minor, uint8_t micro) :
        _version(major,minor,micro) {
        strncpy(_alg, alg, MaxNameSize-1);
    }

    /** Return the packed version (AlgVersion::version()). */
    uint32_t version() {
        return _version.version();
    }

    /** Return the algorithm name buffer. */
    const char* name() {return _alg;}

private:
    char _alg[MaxNameSize];
    AlgVersion _version;
};

/** Description of one data field: an Alg, a name (char[MaxNameSize]), a DataType and a rank. A Names xtc stores an array of these after its header. */
class Name {
public:
    // if you add types here, you must update the corresponding sizes in ShapesData.cc
    // APPEND only: the value is written into the file, so inserting one reinterprets
    // every existing dataset.  FLOAT16 is IEEE binary16, matching CUDA's __half bit
    // for bit, and _Float16 on the host.
    /** Element type of a field. Name::get_element_size() maps each value to a byte size using a table in ShapesData.cc. */
    enum DataType {UINT8, /**< Value 0; element size sizeof(uint8_t). */ UINT16, /**< Value 1; element size sizeof(uint16_t). */ UINT32, /**< Value 2; element size sizeof(uint32_t). */ UINT64, /**< Value 3; element size sizeof(uint64_t). */ INT8, /**< Value 4; element size sizeof(int8_t). */ INT16, /**< Value 5; element size sizeof(int16_t). */ INT32, /**< Value 6; element size sizeof(int32_t). */ INT64, /**< Value 7; element size sizeof(int64_t). */ FLOAT, /**< Value 8; element size sizeof(float). */ DOUBLE, /**< Value 9; element size sizeof(double). */ CHARSTR, /**< Value 10; element size sizeof(char). */ ENUMVAL, /**< Value 11; element size sizeof(int32_t). */ ENUMDICT, /**< Value 12; element size sizeof(int32_t). */ FLOAT16 /**< Value 13; IEEE binary16, element size sizeof(_Float16). */ };

    /** Return the element size in bytes for type from the table in ShapesData.cc; type is not range-checked. */
    static int get_element_size(DataType type);

    /**
     * Field with the given name, type and rank, and an empty Alg ("", 0, 0, 0).
     * Aborts with a message if rank >= MaxRank, if strlen(name) >= MaxNameSize, or if name has a character other than letters, digits, '_', '.' or ':'.
     */
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

    /**
     * Field with the given name, type, rank and alg.
     * Aborts with a message if rank >= MaxRank, if strlen(name) >= MaxNameSize, or if name has a character other than letters, digits, '_', '.' or ':'.
     */
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

    /**
     * Field with the given name and alg, type UINT8 and rank 1.
     * Aborts with a message if strlen(name) >= MaxNameSize or if name has a character other than letters, digits, '_', '.' or ':'.
     */
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

    /** Return the field name. */
    const char* name() {return _name;}
    /** Return the DataType. */
    DataType    type() {return (DataType)_type;}
    /** Return the rank. */
    uint32_t    rank() {return _rank;}
    /** Return a reference to the Alg. */
    Alg&        alg()  {return _alg;}
    /** Return the DataType as a string (e.g. "UINT8"), or nullptr for a value outside the enum. */
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



/** Fixed array of MaxRank uint32_t dimensions. */
class Shape
{
public:
  /** Copy MaxRank entries from shape. */
  Shape(uint32_t shape[MaxRank])
    {
        memcpy(_shape, shape, sizeof(uint32_t) * MaxRank);
    }
    /** Return the product of the first rank dimensions (1 when rank is 0). */
    unsigned num_elements(unsigned rank) {
        unsigned n = 1;
        for (unsigned i = 0; i < rank; i++) {
            n *= _shape[i];
        }
        return n;
    }

    /** Return num_elements(name.rank()) * Name::get_element_size(name.type()), a size in bytes. */
    unsigned size(Name& name) {
        return num_elements(name.rank())*Name::get_element_size(name.type());
    }

    /** Return a pointer to the dimension array. */
    uint32_t* shape() {return _shape;}
private:
    uint32_t _shape[MaxRank]; // in an ideal world this would have variable length "rank"
};

// this class updates the "parent" Xtc extent at the same time
// the child Shapes or Data extent is increased.  the hope is that this
// will make management of the Xtc's less error prone, at the price
// of some performance (more calls to Xtc::alloc())
/** Xtc whose alloc() overloads also add the same size to the extent of a parent (and optionally a grandparent) Xtc. */
class AutoParentAlloc : public Xtc
{
public:
    /** Construct as Xtc(typeId): damage 0, extent sizeof(Xtc). */
    AutoParentAlloc(TypeId typeId) : Xtc(typeId) {}
    /** Construct as Xtc(typeId, namesId): src is set to namesId, damage 0, extent sizeof(Xtc). */
    AutoParentAlloc(TypeId typeId, const NamesId& namesId) : Xtc(typeId,namesId) {}
    /**
     * Add size to parent's extent (parent.alloc()) and to this xtc's extent (Xtc::alloc()), returning the address that was this->next().
     * Either call aborts if bufEnd is non-null and would be passed.
     */
    void* alloc(uint32_t size, Xtc& parent, const void* bufEnd) {
        parent.alloc(size, bufEnd);
        return Xtc::alloc(size, bufEnd);
    }
    /**
     * Add size to the extents of superparent, parent and this xtc (in that order), returning the address that was this->next().
     * Each call aborts if bufEnd is non-null and would be passed.
     */
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
    uint32_t numArrays;  ///< Number of array fields; Names::add() increments it for each added Name with rank > 0.
    char     detType[MaxNameSize];  ///< Copy of the constructor's dettype argument, truncated to MaxNameSize-1 characters and NUL-terminated.
    char     detName[MaxNameSize];  ///< Copy of the constructor's detname argument, truncated to MaxNameSize-1 characters and NUL-terminated.
    char     detId[MaxNameSize];  ///< Copy of the constructor's detid argument, truncated to MaxNameSize-1 characters and NUL-terminated.
    Alg      alg;  ///< Copy of the constructor's alg0 argument.
    uint32_t segment;  ///< Copy of the constructor's segment0 argument.

    /** Copy alg0, segment0 and numarr into the fields, and copy detname, dettype and detid truncated to MaxNameSize-1 characters and NUL-terminated. */
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


/** Xtc of type TypeId::Names: the header, a NameInfo, then an array of Name entries appended by add(). Its src holds the NamesId. */
class Names : public AutoParentAlloc
{
public:

    /**
     * Build the header (TypeId::Names version 0, src = namesId), fill the NameInfo, and grow this xtc's own extent to cover the NameInfo.
     * Aborts if bufEnd is non-null and would be passed, or if detName has a character other than letters, digits or '_'.
     */
    Names(const void* bufEnd, const char* detName, Alg& alg, const char* detType, const char* detId, const NamesId& namesId, unsigned segment=0) :
        AutoParentAlloc(TypeId(TypeId::Names,0),namesId),
        _NameInfo(detName, alg, detType, detId, segment)
    {
        // allocate space for our private data
        Xtc::alloc(sizeof(*this)-sizeof(AutoParentAlloc), bufEnd);
        _checkname();
    }

    /** Return src reinterpreted as a NamesId reference. */
    NamesId& namesId() {return (NamesId&)src;}

    /** Return NameInfo::numArrays. */
    uint32_t numArrays(){return _NameInfo.numArrays;};
    /** Return the detName string from the NameInfo. */
    const char* detName() {return _NameInfo.detName;}
    /** Return the detType string from the NameInfo. */
    const char* detType() {return _NameInfo.detType;}
    /** Return the detId string from the NameInfo. */
    const char* detId()   {return _NameInfo.detId;}
    /** Return the segment number from the NameInfo. */
    unsigned    segment() {return _NameInfo.segment;}
    /** Return a reference to the Alg in the NameInfo. */
    Alg&        alg()     {return _NameInfo.alg;}

    /** Return the index-th Name stored right after this object (no bounds check). */
    Name& get(unsigned index)
    {
        Name& name = ((Name*)(this + 1))[index];
        return name;
    }

    /**
     * Return the number of Name entries: the bytes between the end of this object and next(), divided by sizeof(Name).
     * Aborts with a message if that byte count is not a multiple of sizeof(Name).
     */
    unsigned num()
    {
        unsigned sizeOfNames = (char*)next()-(char*)(this+1);
        if (sizeOfNames%sizeof(Name)!=0) {
            printf("*** %s:%d: Name object alignment error %u\n",__FILE__,__LINE__,unsigned(sizeOfNames%sizeof(Name)));
            abort();
        }
        return sizeOfNames / sizeof(Name);
    }


    /**
     * Append a copy of every Name in V.NameVec, adding sizeof(Name) to this xtc's and parent's extents for each (AutoParentAlloc::alloc()).
     * Increments numArrays for each added Name with rank > 0.
     */
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

/** Child xtc of ShapesData with type TypeId::Data (version 0). */
class Data : public AutoParentAlloc
{
public:
    /** Build a TypeId::Data header (extent sizeof(Xtc)) and add sizeof(Data) to superparent's extent via superparent.alloc(); aborts if bufEnd would be passed. */
    Data(Xtc& superparent, const void* bufEnd) :
        AutoParentAlloc(TypeId(TypeId::Data,0))
    {
        // go two levels up to "auto-alloc" Data Xtc header size
        superparent.alloc(sizeof(*this), bufEnd);
    }
};

/** Child xtc of ShapesData with type TypeId::Shapes (version 0); an array of Shape entries follows the header (see get()). */
class Shapes : public AutoParentAlloc
{
public:
    /**
     * Build a TypeId::Shapes header, add sizeof(Shapes)-sizeof(AutoParentAlloc) (zero, as there is no private data) to its own extent, and add sizeof(Shapes) to superparent's extent.
     * Aborts if bufEnd is non-null and would be passed.
     */
    Shapes(Xtc& superparent, const void* bufEnd) :
        AutoParentAlloc(TypeId(TypeId::Shapes,0))
    {
        // allocate space for our private data
        // not strictly necessary since we currently have no private data.
        Xtc::alloc(sizeof(*this)-sizeof(AutoParentAlloc), bufEnd);
        // go two levels up to "auto-alloc" Shapes size
        superparent.alloc(sizeof(*this), bufEnd);
    }

    /** Return the index-th Shape stored right after this header (no bounds check). */
    Shape& get(unsigned index)
    {
        Shape& shape = ((Shape*)(this + 1))[index];
        return shape;
    }
};

/** Xtc of type TypeId::ShapesData whose payload holds a Shapes xtc and a Data xtc, in either order. Its src holds the NamesId used to find the matching Names. */
class ShapesData : public Xtc
{
public:
    /** Header with type TypeId::ShapesData version 0 and src = namesId; extent sizeof(Xtc), damage 0. */
    ShapesData(NamesId& namesId) : Xtc(TypeId(TypeId::ShapesData,0),namesId) {}

    /** Return src reinterpreted as a NamesId reference. */
    NamesId& namesId() {return (NamesId&)src;}

    /**
     * Return the Data child: the second child xtc if the first one is Shapes, otherwise the first.
     * Aborts with a message if the chosen child's TypeId is not Data.
     */
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

    /**
     * Return the Shapes child: the first child xtc if it is Shapes, otherwise the second.
     * In the second case aborts with a message if that child's TypeId is not Shapes.
     */
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
