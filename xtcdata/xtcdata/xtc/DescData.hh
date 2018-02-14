

#ifndef DESCDATA__H
#define DESCDATA__H

#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/Xtc.hh"
#include <string>
#include <map>
#include <type_traits>
#include <iostream>

#define _unused(x) ((void)(x))


namespace XtcData
{

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
    // template <typename T>
    // typename std::enable_if<is_vec<T>::value, T>::type get_value(const char* namestring)
      
    // {
    //     Data& data = _shapesdata.data();
    //     unsigned index = _nameindex.nameMap()[namestring];
    //     Name& name = _nameindex.names().get(index);
    //     unsigned shapeIndex = _nameindex.shapeMap()[name.name()];
    //     Shape& shape = _shapesdata.shapes().get(shapeIndex);
    //     T array(data.payload() + _offset[index]);
    //     array._shape.resize(name.rank());
    //     for (unsigned i = 0; i < name.rank(); i++) {
    //         array._shape[i] = shape.shape()[i];
    //     }
    //     return array;
    // }

    NameIndex&  nameindex()  {return _nameindex;}
    ShapesData& shapesdata() {return _shapesdata;}

protected:
    // creating a new ShapesData to be filled in
    DescData(NameIndex& nameindex, Xtc& parent) :
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

  void set_array_shape(unsigned index, unsigned shapeIndex, unsigned shape[Name::MaxRank]) {
   
        unsigned rank = _nameindex.names().get(index).rank();

	
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
    DescribedData(Xtc& parent, NameIndex& nameindex, unsigned namesId) :
        DescData(nameindex, parent), _parent(parent), _namesId(namesId)
    {
        new (&_shapesdata) Data(_parent);
    }

  DescribedData(Xtc& parent, std::vector<NameIndex>& NamesVec, unsigned namesId) :
        DescData(NamesVec[namesId], parent), _parent(parent), _namesId(namesId)
    {
        new (&_shapesdata) Data(_parent);
    }
  
    void* data() {return _shapesdata.data().payload();}

    void set_data_length(unsigned size) {
        // now that data has arrived manually update with the number of bytes received
        _shapesdata.data().alloc(size, _shapesdata, _parent);
    }

    void set_array_shape(unsigned index, unsigned shape[Name::MaxRank]) {
        if (_numarrays==0) {
            // add the xtc that will hold the shapes of arrays
            Shapes& shapes = *new (&_shapesdata) Shapes(_parent, _namesId);
            shapes.alloc(_nameindex.shapeMap().size()*sizeof(Shape),
                         _shapesdata, _parent);
        }
	unsigned shapeIndex = _numarrays;
        DescData::set_array_shape(index, shapeIndex, shape);
    }
private:
    Xtc& _parent;
    unsigned      _namesId;
};

class CreateData : public DescData {
public:

  CreateData(Xtc& parent, std::vector<NameIndex>& NamesVec, unsigned namesId) :
      //replaced namesindex with namesvec[namesId]
        DescData(NamesVec[namesId], parent), _parent(parent)
    {
        Shapes& shapes = *new (&_shapesdata) Shapes(_parent, namesId);
        Names& names = _nameindex.names();
        shapes.alloc(names.numArrays()*sizeof(Shape), _shapesdata, _parent);
        new (&_shapesdata) Data(_parent);
    }


  static void check(uint8_t val, Name& name) {
    assert(Name::UINT8==name.type());                                               
  }
  static void check(uint16_t val, Name& name) {
    assert(Name::UINT16==name.type());                                               
  }
  static void check(uint32_t val, Name& name) {
    assert(Name::UINT32==name.type());                                               
  }
  static void check(uint64_t val, Name& name) {
    assert(Name::UINT64==name.type());                                               
  }
  static void check(int8_t val, Name& name) {
    assert(Name::INT8==name.type());                                               
  }
  static void check(int16_t val, Name& name) {
    assert(Name::INT16==name.type());                                               
  }
  static void check(int32_t val, Name& name) {
    assert(Name::INT32==name.type());                                               
  }
  static void check(int64_t val, Name& name) {
    assert(Name::INT64==name.type());                                               
  }
  static void check(float val, Name& name) {
    assert(Name::FLOAT==name.type());                                               
  }  
  static void check(double val, Name& name) {
    assert(Name::DOUBLE==name.type());                                               
  }

  
    
    template <typename T>
    void set_value(unsigned index, T val)
    {
        Data& data = _shapesdata.data();

	if(index != _numentries)
	  {
	   char error_string [100];
	   const char * error_it_name = _nameindex.names().get(index).name();
	     
	   snprintf(error_string,100, "Item \"%s\" with index %d out of order",error_it_name, index);
	   throw std::runtime_error(error_string);
	  }
	
	Name& name = _nameindex.names().get(index);

	check(val, name);
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


  void set_array_shape(unsigned index,unsigned shape[Name::MaxRank]) {
        unsigned int shapeIndex = _numarrays;
	
        assert (shapeIndex==_numarrays);
        _numentries++;
        DescData::set_array_shape(index, shapeIndex, shape);
        Names& names = _nameindex.names();
        Name& namecl = names.get(index);
        unsigned size = _shapesdata.shapes().get(shapeIndex).size(namecl);
        _offset[_numentries]=_offset[_numentries-1]+size;
        _shapesdata.data().alloc(size,_shapesdata,_parent);
    }
  
  

private:
    Xtc& _parent;
};
};
#endif // DESCDATA__H



	
