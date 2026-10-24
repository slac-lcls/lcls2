/**
 * @file
 * @brief Declares DescData (reads fields of a ShapesData using its NameIndex) and the writers DescribedData and CreateData, plus the DESC_* helper macros.
 */
#ifndef DESCDATA__H
#define DESCDATA__H

#include "xtcdata/xtc/Array.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/NamesLookup.hh"
#include "xtcdata/xtc/NameIndex.hh"

#include <string>
#include <type_traits>

/** Cast x to void to silence unused-variable warnings. */
#define _unused(x) ((void)(x))

/** Declare a local XtcData::DescData named oDescData built from itero.method() and names_map[itero.method().namesId()]. */
#define DESC_FOR_METHOD(oDescData, itero, names_map, method) \
        XtcData::DescData oDescData(itero.method(), names_map[itero.method().namesId()])
/** DESC_FOR_METHOD with method shape: declare oDescData from itero.shape(). */
#define DESC_SHAPE(oDescData, itero, names_map) DESC_FOR_METHOD(oDescData, itero, names_map, shape)
/** DESC_FOR_METHOD with method value: declare oDescData from itero.value(). */
#define DESC_VALUE(oDescData, itero, names_map) DESC_FOR_METHOD(oDescData, itero, names_map, value)

namespace XtcData
{

class VarDef;

// this "described data" class glues together the ShapesData
// with the names (including shapes) to compute offsets.
/**
 * Reader for a ShapesData: combines it with the NameIndex of its Names to compute the byte offset of each field in the Data payload.
 * Field offsets are cumulative: a scalar takes Name::get_element_size() bytes and an array takes Shape::size() of the next Shape entry.
 */
class DescData {
public:
    // reading an existing ShapesData
    /**
     * Compute the field offsets for reading an existing shapesdata. Array fields use the Shapes entries in order (0, 1, ...).
     * Throws (from NameIndex::names()) if nameindex holds no Names.
     */
    DescData(ShapesData& shapesdata, NameIndex& nameindex) :
        _offset(nameindex.names().num()+1),
        _shapesdata(shapesdata),
        _nameindex(nameindex),
        _numarrays(0)
    {
        Names& names = _nameindex.names();
        _unused(names);
        _offset[0]=0;
        _numentries = names.num();
        unsigned shapeIndex = 0;
        for (unsigned i=0; _numentries && i<_numentries-1; i++) {
            Name& name = names.get(i);
            if (name.rank()==0) _offset[i+1]=_offset[i]+Name::get_element_size(name.type());
            else {
                // Since we are enforcing consecutive shapes, there's no need for this map lookup
                //unsigned shapeIndex = _nameindex.shapeMap()[name.name()];
                unsigned size = _shapesdata.shapes().get(shapeIndex).size(name);
                _offset[i+1]=_offset[i]+size;
                _numarrays++;
                shapeIndex++;
            }
        }
    }
    /**
     * Copy offsets and counts from o. The ShapesData and NameIndex members are references, so o's objects are assigned into the referenced ones (Xtc::operator= resets that ShapesData's extent to sizeof(Xtc)) instead of rebinding.
     * @return *this.
     */
    DescData& operator=(const DescData& o) { 
        _offset     = o._offset;
        _shapesdata = o._shapesdata;
        _numentries = o._numentries;
        _nameindex  = o._nameindex;
        _numarrays  = o._numarrays;
        return *this;
    }

    /** Destructor; does nothing. */
    ~DescData() {}

    /** Print file, line, and the type and name of name, then call abort(). */
    static void incorrectType(const char* file, unsigned line, Name& name) {
            printf("*** %s:%d: incorrect type %d for %s\n",file,line,name.type(),name.name());
            abort();
    }

    /** Abort (incorrectType()) unless name.type() is UINT8; val only selects the overload. */
    static void checkType(uint8_t val, Name& name) {
        if (Name::UINT8!=name.type()) {
            incorrectType(__FILE__,__LINE__,name);
        }
    }
    /** Abort (incorrectType()) unless name.type() is UINT16; val only selects the overload. */
    static void checkType(uint16_t val, Name& name) {
        if (Name::UINT16!=name.type()) {
            incorrectType(__FILE__,__LINE__,name);
        }
    }
    /** Abort (incorrectType()) unless name.type() is UINT32, ENUMVAL or ENUMDICT; val only selects the overload. */
    static void checkType(uint32_t val, Name& name) {
        if (Name::UINT32!=name.type() && Name::ENUMVAL!=name.type() && Name::ENUMDICT!=name.type()) {
            incorrectType(__FILE__,__LINE__,name);
        }
    }
    /** Abort (incorrectType()) unless name.type() is UINT64; val only selects the overload. */
    static void checkType(uint64_t val, Name& name) {
        if (Name::UINT64!=name.type()) {
            incorrectType(__FILE__,__LINE__,name);
        }
    }
    /** Abort (incorrectType()) unless name.type() is INT8; val only selects the overload. */
    static void checkType(int8_t val, Name& name) {
        if (Name::INT8!=name.type()) {
            incorrectType(__FILE__,__LINE__,name);
        }
    }
    /** Abort (incorrectType()) unless name.type() is INT16; val only selects the overload. */
    static void checkType(int16_t val, Name& name) {
        if (Name::INT16!=name.type()) {
            incorrectType(__FILE__,__LINE__,name);
        }
    }
    /** Abort (incorrectType()) unless name.type() is INT32, ENUMVAL or ENUMDICT; val only selects the overload. */
    static void checkType(int32_t val, Name& name) {
        if (Name::INT32!=name.type() && Name::ENUMVAL!=name.type() && Name::ENUMDICT!=name.type()) {
            incorrectType(__FILE__,__LINE__,name);
        }
    }
    /** Abort (incorrectType()) unless name.type() is INT64; val only selects the overload. */
    static void checkType(int64_t val, Name& name) {
        if (Name::INT64!=name.type()) {
            incorrectType(__FILE__,__LINE__,name);
        }
    }
    /** Abort (incorrectType()) unless name.type() is FLOAT; val only selects the overload. */
    static void checkType(float val, Name& name) {
        if (Name::FLOAT!=name.type()) {
            incorrectType(__FILE__,__LINE__,name);
        }
    }
    /** Abort (incorrectType()) unless name.type() is DOUBLE; val only selects the overload. */
    static void checkType(double val, Name& name) {
        if (Name::DOUBLE!=name.type()) {
            incorrectType(__FILE__,__LINE__,name);
        }
    }
    /** Abort (incorrectType()) unless name.type() is CHARSTR; val only selects the overload. */
    static void checkType(char val, Name& name) {
        if (Name::CHARSTR!=name.type()) {
            incorrectType(__FILE__,__LINE__,name);
        }
    }
    static void checkType(_Float16 val, Name& name) {
        if (Name::FLOAT16!=name.type()) {
            incorrectType(__FILE__,__LINE__,name);
        }
    }


    /**
     * Return an Array<T> view of field index: data at the field's offset in the Data payload, shape from shape(name), rank from the Name.
     * T is not checked against the field's DataType.
     */
    template <typename T>
    Array<T> get_array(unsigned index)
    {
        Name& name = _nameindex.names().get(index);
        uint32_t *shape = this->shape(name);
        Data& data = _shapesdata.data();
        T* ptr = reinterpret_cast<T*>(data.payload() + _offset[index]);

        // Create an Array<T> struct at the memory address of ptr
        Array<T> arrT(ptr, shape, name.rank());
        return arrT;
    };

    // a slower interface to access some data, because
    // it looks it up in the nameMap.
    /** Look up name in the NameIndex's nameMap and return get_value<T>(index). Prints a message and aborts if the name is not found. */
    template <class T>
    T get_value(const char* name)
    {
        IndexMap& nameMap = _nameindex.nameMap();
        if (nameMap.find(name) == nameMap.end()) {
            printf("*** %s:%d: failed to find name %s\n",__FILE__,__LINE__,name);
            abort();
        }
        unsigned index = nameMap[name];

        return get_value<T>(index);
    }

    /**
     * Return a copy of the T stored at field index's offset in the Data payload.
     * Aborts with a message if index > the number of entries (index equal to it is not rejected), or via checkType() if T does not match the field's DataType.
     */
    template <class T>
    T get_value(unsigned index)
    {
        if (index > _numentries) {
            printf("*** %s:%d: index %d out of range %d\n",__FILE__,__LINE__,index,_numentries);
            abort();
        }
        Data& data = _shapesdata.data();
        Name& name = _nameindex.names().get(index);

        T val = *reinterpret_cast<T*>(data.payload() + _offset[index]);
        checkType(val, name);
        return val;
    }

    /**
     * Return a reference to the T at field index's offset in the Data payload.
     * Aborts with a message if index > the number of entries; the type is not checked.
     */
    template <class T>
    T& get_lvalue(unsigned index)
    {
        if (index > _numentries) {
            printf("*** %s:%d: index %d out of range %d\n",__FILE__,__LINE__,index,_numentries);
            abort();
        }
        Data& data = _shapesdata.data();
        Name& name = _nameindex.names().get(index);

        T& val = *reinterpret_cast<T*>(data.payload() + _offset[index]);
        return val;
    }

    // void* address(unsigned index) {
    //     Data& data = _shapesdata.data();
    //     return data.payload() + _offset[index];
    // }

    /** Return the dimension array of the Shape at index shapeMap()[name.name()] in the Shapes child. A name missing from shapeMap is inserted with index 0 by std::map::operator[]. */
    uint32_t* shape(Name& name) {
        Shapes& shapes = _shapesdata.shapes();
        unsigned shapeIndex = _nameindex.shapeMap()[name.name()];
        return shapes.get(shapeIndex).shape();
    }

    /** Return the NameIndex reference. */
    NameIndex&  nameindex()  {return _nameindex;}
    /** Return the ShapesData reference. */
    ShapesData& shapesdata() {return _shapesdata;}
    /** Assign o into the referenced ShapesData (Xtc::operator=, which copies src, damage and contains and resets extent to sizeof(Xtc)); the reference is not rebound. */
    void        shapesdata(ShapesData& o) { _shapesdata=o; }
protected:
    // creating a new ShapesData to be filled in
    DescData(NameIndex& nameindex, Xtc& parent, const void* bufEnd, NamesId& namesId) :
        _offset(nameindex.names().num()+1),
        _shapesdata(*new (parent, bufEnd) ShapesData(namesId)),
        _nameindex(nameindex),
        _numarrays(0)
    {
        Names& names = _nameindex.names();
        _unused(names);
        _offset[0]=0;
        _numentries=0;
    }

    DescData(NameIndex& nameindex, Xtc& parent, const void* bufEnd, VarDef& V, NamesId& namesId) :
        _offset(nameindex.names().num()+1),
        _shapesdata(*new (parent, bufEnd) ShapesData(namesId)),
        _nameindex(nameindex),
        _numarrays(0)
    {
        Names& names = _nameindex.names();
        _unused(names);
        _offset[0]=0;
        _numentries=0;
    }
    void set_array_shape(unsigned index, unsigned shapeIndex, const unsigned shape[MaxRank]) {

        unsigned rank = _nameindex.names().get(index).rank();

        if (rank==0) {
            printf("*** %s:%d: can't set_array_shape for scalers\n",__FILE__,__LINE__);
            abort();
        }
        if (shapeIndex!=_numarrays) {
            printf("*** %s:%d: array filled out of order\n",__FILE__,__LINE__);
            abort();
        }
        _unused(shapeIndex);
        Shape& sh = _shapesdata.shapes().get(_numarrays);
        for (unsigned i=0; i<rank; i++) {
            sh.shape()[i] = shape[i];
        }
        _numarrays++;
    }


    std::vector<unsigned> _offset;
    ShapesData& _shapesdata;
    unsigned    _numentries;
    NameIndex&  _nameindex;
    unsigned    _numarrays;
};

/** Writer that appends a ShapesData with an empty Data child; the caller writes the payload at data() and then records its length with set_data_length(). Array shapes are added with set_array_shape(). */
class DescribedData : public DescData {
public:
    /**
     * Append to parent a ShapesData (src namesId) and a Data child inside it, growing the extents of parent and the ShapesData.
     * Throws (from NameIndex::names()) if nameindex holds no Names.
     */
    DescribedData(Xtc& parent, const void* bufEnd, NameIndex& nameindex, NamesId& namesId) :
        DescData(nameindex, parent, bufEnd, namesId), _parent(parent), _bufEnd(bufEnd)
    {
        new (&_shapesdata, bufEnd) Data(_parent, bufEnd);
    }

    /** Same as the NameIndex constructor, using NamesLookup[namesId] as the NameIndex (a missing key is inserted empty and names() then throws). */
    DescribedData(Xtc& parent, const void* bufEnd, NamesLookup& NamesLookup, NamesId& namesId) :
        DescData(NamesLookup[namesId], parent, bufEnd, namesId), _parent(parent), _bufEnd(bufEnd)
    {
        new (&_shapesdata, bufEnd) Data(_parent, bufEnd);
    }

    /** Return a pointer to the start of the Data child's payload. */
    void* data() {return _shapesdata.data().payload();}

    /** Add size bytes to the extents of the Data child, the ShapesData and parent (AutoParentAlloc::alloc()). The source comment says this is done once the data has arrived. */
    void set_data_length(unsigned size) {
        // now that data has arrived manually update with the number of bytes received
        _shapesdata.data().alloc(size, _shapesdata, _parent, _bufEnd);
    }

    /**
     * Store shape as the next Shape entry for array field index. On the first call it first appends a Shapes child with room for shapeMap().size() Shape entries.
     * Aborts if field index is a scalar (rank 0).
     */
    void set_array_shape(unsigned index, const unsigned shape[MaxRank]) {
        if (_numarrays==0) {
            // add the xtc that will hold the shapes of arrays
            Shapes& shapes = *new (&_shapesdata, _bufEnd) Shapes(_parent, _bufEnd);
            shapes.alloc(_nameindex.shapeMap().size()*sizeof(Shape),
                         _shapesdata, _parent, _bufEnd);
        }
        unsigned shapeIndex = _numarrays;
        DescData::set_array_shape(index, shapeIndex, shape);
    }
private:
    Xtc&        _parent;
    const void* _bufEnd;
};

/**
 * Writer that builds a ShapesData field by field: the constructor appends the ShapesData, a Shapes child with room for numArrays() Shape entries, and an empty Data child.
 * set_value(), allocate() and set_string() then append fields; the destructor aborts if the number written differs from the Names count.
 */
class CreateData : public DescData {
public:
    /**
     * Append to parent a ShapesData (src namesId) holding a Shapes child with room for names.numArrays() Shape entries and an empty Data child, using NamesLookup[namesId] as the NameIndex.
     * Throws (from NameIndex::names()) if that entry holds no Names; a missing key is inserted empty.
     */
    CreateData(Xtc& parent, const void* bufEnd, NamesLookup& NamesLookup, NamesId& namesId) :
        DescData(NamesLookup[namesId], parent, bufEnd, namesId), _parent(parent), _bufEnd(bufEnd)
    {
        Shapes& shapes = *new (&_shapesdata, _bufEnd) Shapes(_parent, _bufEnd);
        Names& names = _nameindex.names();
        _numExpectedEntries = names.num();
        shapes.alloc(names.numArrays()*sizeof(Shape), _shapesdata, _parent, _bufEnd);
        new (&_shapesdata, _bufEnd) Data(_parent, _bufEnd);
    }

    /** Same as the four-argument constructor; V is not used. */
    CreateData(Xtc& parent, const void* bufEnd, NamesLookup& NamesLookup, VarDef& V, NamesId& namesId) :
        DescData(NamesLookup[namesId], parent, bufEnd, V, namesId), _parent(parent), _bufEnd(bufEnd)
    {
        Shapes& shapes = *new (&_shapesdata, _bufEnd) Shapes(_parent, _bufEnd);
        Names& names = _nameindex.names();
        _numExpectedEntries = names.num();
        shapes.alloc(names.numArrays()*sizeof(Shape), _shapesdata, _parent, _bufEnd);
        new (&_shapesdata, _bufEnd) Data(_parent, _bufEnd);
    }

    /** Print a message and call abort() if the number of fields written differs from the number of Name entries in the Names. */
    ~CreateData()
    {
        if (_numentries != _numExpectedEntries) {
            printf("CreateData: %d entries not equal to number of expected entries %d\n", _numentries, _numExpectedEntries);
            abort();
        }
    }

    /**
     * Reserve array field index at the end of the Data payload and return an Array<T> view over it; the view's shape pointer is the caller's shape array.
     * Calls set_array_shape(index, shape), which stores the shape and grows the extents by the array's byte size. Aborts (checkType()) if T does not match the field's DataType; the field order is not checked.
     */
    template <typename T>
    Array<T> allocate(unsigned index, unsigned *shape)
    {
        Name& name = _nameindex.names().get(index);
        T val = '\0'; checkType(val, name);

        //Create a pointer to the next part of contiguous memory
        void *ptr = reinterpret_cast<void *>(_shapesdata.data().next());

        // Create an Array<T> struct at the memory address of ptr
        Array<T> arrT(ptr, shape, name.rank());
        CreateData::set_array_shape(index, shape);

        // Return the Array struct. Use it to assign values with arrayT(i,j)
        return arrT;
    };

    /**
     * Append field index as a CHARSTR array holding xtcstring and its NUL terminator.
     * The array length is strlen + 1 rounded up to a multiple of 4, capped at the bytes left before bufEnd; aborts (checkType()) if the field is not CHARSTR.
     */
    void set_string(unsigned index, const char* xtcstring)
    {
        // include the null character
        unsigned bytes = strlen(xtcstring)+1;
        // allocate in units of 4 bytes, to do some reasonable alignment
        // although maybe this doesn't make sense since uint8_t arrays
        // can have any length
        bytes = ((bytes-1)/4)*4+4;
        // protect against being passed an un-terminated string
        auto MaxStrLen = (reinterpret_cast<const char *>(_bufEnd) -
                          reinterpret_cast<char *>(_shapesdata.data().next()));
        if (bytes>MaxStrLen) bytes = MaxStrLen;
        unsigned charStrShape[MaxRank];
        charStrShape[0] = bytes;
        Array<char> charArray = allocate<char>(index,charStrShape);
        // strncat(): string in dest is always null-terminated.
        *(charArray.data()) = '\0';
        strncat(charArray.data(),xtcstring,MaxStrLen-1);
    }

    /**
     * Write val as scalar field index at the end of the Data payload and grow the Data, ShapesData and parent extents by sizeof(T).
     * Aborts with a message if index is not the next field in order, or (checkType()) if T does not match the field's DataType.
     */
    template <typename T>
    void set_value(unsigned index, T val)
    {
        Data& data = _shapesdata.data();

        if(index != _numentries) {
            const char * error_it_name = _nameindex.names().get(index).name();

            printf("Item \"%s\" with index %d out of order",error_it_name, index);
            abort();
        }

        Name& name = _nameindex.names().get(index);

        checkType(val, name);
        T* ptr = reinterpret_cast<T*>(data.payload() + _offset[index]);
        *ptr = val;
        data.alloc(sizeof(T), _shapesdata, _parent, _bufEnd);
        _numentries++;
        _offset[_numentries]=_offset[_numentries-1]+Name::get_element_size(name.type());
    }

    /** Return the address just past the Data child (its next()), where the next field would be written. */
    void* get_ptr()
    {
        return reinterpret_cast<void*>(_shapesdata.data().next());
    }

    /**
     * Store shape as the next Shape entry for array field index, count the field as written, and grow the Data, ShapesData and parent extents by the array's byte size (Shape::size()).
     * Aborts if field index is a scalar (rank 0).
     */
    void set_array_shape(unsigned index,unsigned shape[MaxRank]) {
        unsigned int shapeIndex = _numarrays;

        if (shapeIndex!=_numarrays) {
            printf("*** %s:%d: array filled out of order\n",__FILE__,__LINE__);
            abort();
        }
        _numentries++;
        DescData::set_array_shape(index, shapeIndex, shape);
        Names& names = _nameindex.names();
        Name& namecl = names.get(index);
        unsigned size = _shapesdata.shapes().get(shapeIndex).size(namecl);
        _offset[_numentries]=_offset[_numentries-1]+size;
        _shapesdata.data().alloc(size,_shapesdata,_parent,_bufEnd);
    }

private:
    unsigned    _numExpectedEntries;
    Xtc&        _parent;
    const void* _bufEnd;
};

}; // namespace XtcData

#endif // DESCDATA__H
