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

#define _unused(x) ((void)(x))

#define DESC_FOR_METHOD(oDescData, itero, names_map, method) \
        XtcData::DescData oDescData(itero.method(), names_map[itero.method().namesId()])
#define DESC_SHAPE(oDescData, itero, names_map) DESC_FOR_METHOD(oDescData, itero, names_map, shape)
#define DESC_VALUE(oDescData, itero, names_map) DESC_FOR_METHOD(oDescData, itero, names_map, value)

namespace XtcData
{

class VarDef;

// this "described data" class glues together the ShapesData
// with the names (including shapes) to compute offsets.
class DescData {
public:
    // reading an existing ShapesData
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
    DescData& operator=(const DescData& o) { 
        _offset     = o._offset;
        _shapesdata = o._shapesdata;
        _numentries = o._numentries;
        _nameindex  = o._nameindex;
        _numarrays  = o._numarrays;
        return *this;
    }

    ~DescData() {}

    static void incorrectType(const char* file, unsigned line, Name& name) {
            printf("*** %s:%d: incorrect type %d for %s\n",file,line,name.type(),name.name());
            abort();
    }

    static void checkType(uint8_t val, Name& name) {
        if (Name::UINT8!=name.type()) {
            incorrectType(__FILE__,__LINE__,name);
        }
    }
    static void checkType(uint16_t val, Name& name) {
        if (Name::UINT16!=name.type()) {
            incorrectType(__FILE__,__LINE__,name);
        }
    }
    static void checkType(uint32_t val, Name& name) {
        if (Name::UINT32!=name.type() && Name::ENUMVAL!=name.type() && Name::ENUMDICT!=name.type()) {
            incorrectType(__FILE__,__LINE__,name);
        }
    }
    static void checkType(uint64_t val, Name& name) {
        if (Name::UINT64!=name.type()) {
            incorrectType(__FILE__,__LINE__,name);
        }
    }
    static void checkType(int8_t val, Name& name) {
        if (Name::INT8!=name.type()) {
            incorrectType(__FILE__,__LINE__,name);
        }
    }
    static void checkType(int16_t val, Name& name) {
        if (Name::INT16!=name.type()) {
            incorrectType(__FILE__,__LINE__,name);
        }
    }
    static void checkType(int32_t val, Name& name) {
        if (Name::INT32!=name.type() && Name::ENUMVAL!=name.type() && Name::ENUMDICT!=name.type()) {
            incorrectType(__FILE__,__LINE__,name);
        }
    }
    static void checkType(int64_t val, Name& name) {
        if (Name::INT64!=name.type()) {
            incorrectType(__FILE__,__LINE__,name);
        }
    }
    static void checkType(float val, Name& name) {
        if (Name::FLOAT!=name.type()) {
            incorrectType(__FILE__,__LINE__,name);
        }
    }
    static void checkType(double val, Name& name) {
        if (Name::DOUBLE!=name.type()) {
            incorrectType(__FILE__,__LINE__,name);
        }
    }
    static void checkType(char val, Name& name) {
        if (Name::CHARSTR!=name.type()) {
            incorrectType(__FILE__,__LINE__,name);
        }
    }


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

    uint32_t* shape(Name& name) {
        Shapes& shapes = _shapesdata.shapes();
        unsigned shapeIndex = _nameindex.shapeMap()[name.name()];
        return shapes.get(shapeIndex).shape();
    }

    NameIndex&  nameindex()  {return _nameindex;}
    ShapesData& shapesdata() {return _shapesdata;}
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

class DescribedData : public DescData {
public:
    DescribedData(Xtc& parent, const void* bufEnd, NameIndex& nameindex, NamesId& namesId) :
        DescData(nameindex, parent, bufEnd, namesId), _parent(parent), _bufEnd(bufEnd)
    {
        new (&_shapesdata, bufEnd) Data(_parent, bufEnd);
    }

    DescribedData(Xtc& parent, const void* bufEnd, NamesLookup& NamesLookup, NamesId& namesId) :
        DescData(NamesLookup[namesId], parent, bufEnd, namesId), _parent(parent), _bufEnd(bufEnd)
    {
        new (&_shapesdata, bufEnd) Data(_parent, bufEnd);
    }

    void* data() {return _shapesdata.data().payload();}

    void set_data_length(unsigned size) {
        // now that data has arrived manually update with the number of bytes received
        _shapesdata.data().alloc(size, _shapesdata, _parent, _bufEnd);
    }

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

class CreateData : public DescData {
public:
    CreateData(Xtc& parent, const void* bufEnd, NamesLookup& NamesLookup, NamesId& namesId) :
        DescData(NamesLookup[namesId], parent, bufEnd, namesId), _parent(parent), _bufEnd(bufEnd)
    {
        Shapes& shapes = *new (&_shapesdata, _bufEnd) Shapes(_parent, _bufEnd);
        Names& names = _nameindex.names();
        _numExpectedEntries = names.num();
        shapes.alloc(names.numArrays()*sizeof(Shape), _shapesdata, _parent, _bufEnd);
        new (&_shapesdata, _bufEnd) Data(_parent, _bufEnd);
    }

    CreateData(Xtc& parent, const void* bufEnd, NamesLookup& NamesLookup, VarDef& V, NamesId& namesId) :
        DescData(NamesLookup[namesId], parent, bufEnd, V, namesId), _parent(parent), _bufEnd(bufEnd)
    {
        Shapes& shapes = *new (&_shapesdata, _bufEnd) Shapes(_parent, _bufEnd);
        Names& names = _nameindex.names();
        _numExpectedEntries = names.num();
        shapes.alloc(names.numArrays()*sizeof(Shape), _shapesdata, _parent, _bufEnd);
        new (&_shapesdata, _bufEnd) Data(_parent, _bufEnd);
    }

    ~CreateData()
    {
        if (_numentries != _numExpectedEntries) {
            printf("CreateData: %d entries not equal to number of expected entries %d\n", _numentries, _numExpectedEntries);
            abort();
        }
    }

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

    void* get_ptr()
    {
        return reinterpret_cast<void*>(_shapesdata.data().next());
    }

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
