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

struct cmp_str
{
   bool operator()(const char *a, const char *b)
   {
      return std::strcmp(a, b) < 0;
   }
};

typedef std::map<const char *, unsigned, cmp_str> IndexMap;

class NameIndex : public IndexMap {
public:
    NameIndex(Names& names) : _names(names) {
        unsigned iarray = 0;
        for (unsigned i=0; i<names.num(); i++) {
            Name& name = _names.get(i);
            (*this)[name.name()]=i;
            if (name.rank()>0) {
                _shapeMap[name.name()]=iarray;
                iarray++;
            }
        }
    }
    IndexMap& shapeMap() {return _shapeMap;}
    Names& names() {return _names;}
private:
    Names&   _names;
    IndexMap _shapeMap;
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
            }
        }
    }

    // all fundamental types
    template <typename T>
    typename std::enable_if<std::is_fundamental<T>::value, T>::type get_value(const char* name)
    {
        Data& data = _shapesdata.data();
        unsigned index = _nameindex[name];
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
        unsigned index = _nameindex[namestring];
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

    enum {MaxNames=1000};
    unsigned    _offset[MaxNames+1]; // +1 since we set up the offsets 1 in advance, for convenience
    ShapesData& _shapesdata;
    unsigned    _numentries;
    NameIndex&  _nameindex;
    unsigned    _numarrays;
};

class FrontEndData : public DescData {
public:
    FrontEndData(XtcData::Xtc& parent, NameIndex& nameindex, unsigned namesId) :
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
        unsigned index = _nameindex[name];
        unsigned rank = _nameindex.names().get(index).rank();
        unsigned shapeIndex = _nameindex.shapeMap()[name];
        assert (shapeIndex==_numarrays);
        _unused(shapeIndex);
        Shape& sh = _shapesdata.shapes().get(_numarrays);
        for (unsigned i=0; i<rank; i++) {
            sh.shape()[i] = shape[i];
        }
        _numarrays++;
    }
private:
    XtcData::Xtc& _parent;
    unsigned      _namesId;
};

class FexData : public DescData {
public:
    FexData(XtcData::Xtc& parent, NameIndex& nameindex, unsigned namesId) :
        DescData(nameindex, parent), _parent(parent)
    {
        Shapes& shapes = *new (&_shapesdata) Shapes(_parent, namesId);
        Names& names = _nameindex.names();
        shapes.alloc(names.num()*sizeof(Shape), _shapesdata, _parent);
        new (&_shapesdata) Data(_parent);
    }

    template <typename T>
    void set_value(const char* namestring, T val)
    {
        Data& data = _shapesdata.data();
        unsigned index = _nameindex[namestring];
        assert (index==_numentries); // require the user to fill the fields in order
        Name& name = _nameindex.names().get(index);
        T* ptr = reinterpret_cast<T*>(data.payload() + _offset[index]);
        *ptr = val;
        data.alloc(sizeof(T), _shapesdata, _parent);
        _numentries++;
        _offset[_numentries]=_offset[_numentries-1]+Name::get_element_size(name.type());
    }
private:
    XtcData::Xtc& _parent;
};

#endif // DESCDATA__H
