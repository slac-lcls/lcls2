#ifndef DESCDATA__H
#define DESCDATA__H

#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/VarDef.hh"

#include <string>
#include <map>
#include <type_traits>
#include <iostream>

#define _unused(x) ((void)(x))
 
namespace XtcData
{

class VarDef;
	       
template <typename T>
class Array {
public:

    Array(void *data, uint32_t *shape, uint32_t rank){
        _shape = shape;
        _data = reinterpret_cast<T*>(data);
        _rank = rank;
    }
    T& operator()(unsigned i){
        assert(i < _shape[0]);
        return _data[i];
    }
    T& operator()(unsigned i, unsigned j){
        assert(i<_shape[0]);assert(j<_shape[1]);
        return _data[i * _shape[1] + j];
    }
    const T& operator()(unsigned i, unsigned j) const{
        assert(i< _shape[0]);assert(j<_shape[1]);
        return _data[i * _shape[1] + j];
    }
    T& operator()(unsigned i, unsigned j, unsigned k){
        assert(i< _shape[0]);assert(j<_shape[1]);assert(k<_shape[3]);
        return _data[(i * _shape[1] + j) * _shape[2] + k];
    }
    const T& operator()(unsigned i, unsigned j, unsigned k) const
    {
        assert(i< _shape[0]);assert(j<_shape[1]);assert(k<_shape[3]);
        return _data[(i * _shape[1] + j) * _shape[2] + k];
    }
    uint32_t rank(){
        return _rank;
    }
    uint32_t* shape(){
        return _shape;
    }
    T* data(){
        return _data;
    }
    uint64_t num_elem(){
        uint64_t _num_elem = _shape[0];
        for(uint32_t i=1; i<_rank;i++){_num_elem*=_shape[i];};
        return _num_elem;
    }
    void shape(uint32_t a, uint32_t b=0, uint32_t c=0, uint32_t d=0, uint32_t e=0){
        assert(_rank > 0);
        assert(XtcData::Name::MaxRank == 5);
        _shape[0] = a;
        _shape[1] = b;
        _shape[2] = c;
        _shape[3] = d;
        _shape[4] = e;
    }

protected:
    uint32_t *_shape;
    T        *_data;
    uint32_t  _rank;
    Array(){}
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
        unsigned shapeIndex = 0;
        for (unsigned i=0; i<_numentries-1; i++) {
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



    // all fundamental types
    // simplify get_value
    // split into templated function based on the return type


    // add get_array here

    
    static void checkType(uint8_t val, Name& name) {
	assert(Name::UINT8==name.type());
    }
    static void checkType(uint16_t val, Name& name) {
	assert(Name::UINT16==name.type());
    }
    static void checkType(uint32_t val, Name& name) {
	assert(Name::UINT32==name.type());
    }
    static void checkType(uint64_t val, Name& name) {
	assert(Name::UINT64==name.type());
    }
    static void checkType(int8_t val, Name& name) {
	assert(Name::INT8==name.type());
    }
    static void checkType(int16_t val, Name& name) {
	assert(Name::INT16==name.type());
    }
    static void checkType(int32_t val, Name& name) {
	assert(Name::INT32==name.type());
    }
    static void checkType(int64_t val, Name& name) {
	assert(Name::INT64==name.type());
    }
    static void checkType(float val, Name& name) {
	assert(Name::FLOAT==name.type());
    }
    static void checkType(double val, Name& name) {
	assert(Name::DOUBLE==name.type());
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

    template <class T>
    T get_value(const char* name)
    {
        Data& data = _shapesdata.data();
        unsigned index = _nameindex.nameMap()[name];

        T val = *reinterpret_cast<T*>(data.payload() + _offset[index]);
        checkType(val, _nameindex.names().get(index));
        return val;
    }

    template <class T>
    T get_value(unsigned index) //, T& val)
    {
        assert(index <= _numentries);
        Data& data = _shapesdata.data();
        Name& name = _nameindex.names().get(index);

        T val = *reinterpret_cast<T*>(data.payload() + _offset[index]);
        checkType(val, name);
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

    DescData(NameIndex& nameindex, Xtc& parent, VarDef& V) :
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

        CreateData(Xtc& parent, std::vector<NameIndex>& NamesVec, unsigned namesId, VarDef& V) :
            //replaced namesindex with namesvec[namesId]
            DescData(NamesVec[namesId], parent, V), _parent(parent)
        {
            Shapes& shapes = *new (&_shapesdata) Shapes(_parent, namesId);
            Names& names = _nameindex.names();
            shapes.alloc(names.numArrays()*sizeof(Shape), _shapesdata, _parent);
            new (&_shapesdata) Data(_parent);
        }


        template <typename T>
        Array<T> allocate(unsigned index, unsigned *shape)
        {
            Name& name = _nameindex.names().get(index);
            T val;checkType(val, name);

            Data& data = _shapesdata.data();
            //Create a pointer to the next part of contiguous memory
            void *ptr = reinterpret_cast<void *>(_shapesdata.data().next());

            // Create an Array<T> struct at the memory address of ptr
            Array<T> arrT(ptr, shape, name.rank());
            this->set_array_shape(index, shape);

            // Return the Array struct. Use it to assign values with arrayT(i,j)
            return arrT;
        };

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

            checkType(val, name);
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
            //	printf("Sizeof size is %i\n", size);
            _offset[_numentries]=_offset[_numentries-1]+size;
            _shapesdata.data().alloc(size,_shapesdata,_parent);
        }



    private:
        Xtc& _parent;
    };
};
#endif // DESCDATA__H




