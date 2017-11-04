#ifndef DESCDATA__H
#define DESCDATA__H

#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/Xtc.hh"
#include <string>
#include <map>
#include <type_traits>

#define _unused(x) ((void)(x))

template <typename T>
struct Array {
    Array(uint8_t* buffer)
    {
        data = reinterpret_cast<T*>(buffer);
    }
    T& operator()(int i, int j)
    {
        return data[i * _shape[1] + j];
    }
    const T& operator()(int i, int j) const
    {
        return data[i * _shape[1] + j];
    }
    T& operator()(int i, int j, int k)
    {
        return data[(i * _shape[1] + j) * _shape[2] + k];
    }
    const T& operator()(int i, int j, int k) const
    {
        return data[(i * _shape[1] + j) * _shape[2] + k];
    }

    T* data;
    std::vector<int> _shape;
};

template <typename>
struct is_vec : std::false_type {
};

template <typename T>
struct is_vec<Array<T>> : std::true_type {
};

typedef std::map<std::string, unsigned> IndexMap;

class NameIndex {
public:
    NameIndex(Names& names) {
        _init_names(names);
        unsigned iarray = 0;
        for (unsigned i=0; i<_names->num(); i++) {
            Name& name = _names->get(i);
            _nameMap[std::string(name.name())]=i;
            if (name.rank()>0) {
                _shapeMap[std::string(name.name())]=iarray;
                iarray++;
            }
        }
    }
    NameIndex(const NameIndex& old) {
        _init_names(*old._names);
        _shapeMap = old._shapeMap;
        _nameMap = old._nameMap;
    }
    NameIndex& operator=(const NameIndex& rhs) = delete;
    ~NameIndex() {delete _names;}
    IndexMap& shapeMap() {return _shapeMap;}
    IndexMap& nameMap()  {return _nameMap;}
    Names& names() {return *_names;}
private:
    void _init_names(Names& names) {
        _names = (Names*)malloc(names.extent);
        std::memcpy(_names, &names, names.extent);
    }
    Names*   _names;
    IndexMap _shapeMap;
    IndexMap _nameMap;
};

// this "described data" class glues together the ShapesData
// with the names (including shapes) to compute offsets.
class DescData {
public:
    // reading an existing ShapesData
    DescData(ShapesData& shapesdata, NameIndex& nameindex) :
        _shapesdata(shapesdata),
        _nameindex(nameindex),
        _numarrays(0)
    {
        Names& names = _nameindex.names();
        assert(names.num()<MaxNames);
        _unused(names);
        _offset[0]=0;
        _numentries = names.num();
        for (unsigned i=0; i<_numentries-1; i++) {
            Name& name = names.get(i);
            if (name.rank()==0) _offset[i+1]=_offset[i]+Name::get_element_size(name.type());
            else {
                unsigned shapeIndex = _nameindex.shapeMap()[name.name()];
                unsigned size = _shapesdata.shapes().get(shapeIndex).size(name);
                _offset[i+1]=_offset[i]+size;
                _numarrays++;
            }
        }
    }

    // all fundamental types
    template <typename T>
    typename std::enable_if<std::is_fundamental<T>::value, T>::type get_value(const char* name)
    {
        Data& data = _shapesdata.data();
        unsigned index = _nameindex.nameMap()[name];
        return *reinterpret_cast<T*>(data.payload() + _offset[index]);
    }

    void* address(unsigned index) {
        Data& data = _shapesdata.data();
        return data.payload() + _offset[index];
    }

    uint32_t* shape(Name& name) {
        Shapes& shapes = _shapesdata.shapes();
        unsigned shapeIndex = _nameindex.shapeMap()[name.name()];
        return shapes.get(shapeIndex).shape();
    }

    // for all array types
    template <typename T>
    typename std::enable_if<is_vec<T>::value, T>::type get_value(const char* namestring)
    {
        Data& data = _shapesdata.data();
        unsigned index = _nameindex.nameMap()[namestring];
        Name& name = _nameindex.names().get(index);
        unsigned shapeIndex = _nameindex.shapeMap()[name.name()];
        Shape& shape = _shapesdata.shapes().get(shapeIndex);
        T array(data.payload() + _offset[index]);
        array._shape.resize(name.rank());
        for (unsigned i = 0; i < name.rank(); i++) {
            array._shape[i] = shape.shape()[i];
        }
        return array;
    }

    NameIndex&  nameindex()  {return _nameindex;}
    ShapesData& shapesdata() {return _shapesdata;}

protected:
    // creating a new ShapesData to be filled in
    DescData(NameIndex& nameindex, XtcData::Xtc& parent) :
        _shapesdata(*new (parent) ShapesData()),
        _nameindex(nameindex),
        _numarrays(0)
    {
        Names& names = _nameindex.names();
        assert(names.num()<MaxNames);
        _unused(names);
        _offset[0]=0;
        _numentries=0;
    }

    void set_array_shape(const char* name, unsigned shape[Name::MaxRank]) {
        unsigned index = _nameindex.nameMap()[name];
        unsigned rank = _nameindex.names().get(index).rank();
        unsigned shapeIndex = _nameindex.shapeMap()[name];
        assert (shapeIndex==_numarrays); // check that shapes are filled in order
        _unused(shapeIndex);
        Shape& sh = _shapesdata.shapes().get(_numarrays);
        for (unsigned i=0; i<rank; i++) {
            sh.shape()[i] = shape[i];
        }
        _numarrays++;
    }

    enum {MaxNames=1000};
    unsigned    _offset[MaxNames+1]; // +1 since we set up the offsets 1 in advance, for convenience
    ShapesData& _shapesdata;
    unsigned    _numentries;
    NameIndex&  _nameindex;
    unsigned    _numarrays;
};

class DescribedData : public DescData {
public:
    DescribedData(XtcData::Xtc& parent, NameIndex& nameindex, unsigned namesId) :
        DescData(nameindex, parent), _parent(parent), _namesId(namesId)
    {
        new (&_shapesdata) Data(_parent);
    }

    void* data() {return _shapesdata.data().payload();}

    void set_data_length(unsigned size) {
        // now that data has arrived manually update with the number of bytes received
        _shapesdata.data().alloc(size, _shapesdata, _parent);
    }

    void set_array_shape(const char* name, unsigned shape[Name::MaxRank]) {
        if (_numarrays==0) {
            // add the xtc that will hold the shapes of arrays
            Shapes& shapes = *new (&_shapesdata) Shapes(_parent, _namesId);
            shapes.alloc(_nameindex.shapeMap().size()*sizeof(Shape),
                         _shapesdata, _parent);
        }
        DescData::set_array_shape(name, shape);
    }
private:
    XtcData::Xtc& _parent;
    unsigned      _namesId;
};

#include <stdio.h>
class CreateData : public DescData {
public:
    CreateData(XtcData::Xtc& parent, NameIndex& nameindex, unsigned namesId) :
        DescData(nameindex, parent), _parent(parent)
    {
        Shapes& shapes = *new (&_shapesdata) Shapes(_parent, namesId);
        Names& names = _nameindex.names();
        // this wastes space: should be arrays.num
        shapes.alloc(names.num()*sizeof(Shape), _shapesdata, _parent);
        new (&_shapesdata) Data(_parent);
    }

    template <typename T>
    void set_value(const char* namestring, T val)
    {
        Data& data = _shapesdata.data();
        unsigned index = _nameindex.nameMap()[namestring];
        assert (index==_numentries); // require the user to fill the fields in order
        Name& name = _nameindex.names().get(index);
        T* ptr = reinterpret_cast<T*>(data.payload() + _offset[index]);
        *ptr = val;
        data.alloc(sizeof(T), _shapesdata, _parent);
        _numentries++;
        _offset[_numentries]=_offset[_numentries-1]+Name::get_element_size(name.type());
    }

    void* get_ptr()
    {
        return reinterpret_cast<void*>(_shapesdata.data().next());
    }

    void set_array_shape(const char* name, unsigned shape[Name::MaxRank]) {
        unsigned index = _nameindex.nameMap()[name];
        assert (index==_numentries); // require the user to fill the fields in order
        unsigned shapeIndex = _nameindex.shapeMap()[name];
        assert (shapeIndex==_numarrays);
        _numentries++;
        DescData::set_array_shape(name, shape);
        Names& names = _nameindex.names();
        Name& namecl = names.get(index);
        unsigned size = _shapesdata.shapes().get(shapeIndex).size(namecl);
        _offset[_numentries]=_offset[_numentries-1]+size;
        _shapesdata.data().alloc(size,_shapesdata,_parent);
    }

private:
    XtcData::Xtc& _parent;
};

#endif // DESCDATA__H
