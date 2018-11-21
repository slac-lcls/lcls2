#ifndef DESCDATA__H
#define DESCDATA__H

#include "xtcdata/xtc/Array.hh"
#include "xtcdata/xtc/ShapesData.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/NamesVec.hh"
#include "xtcdata/xtc/NameIndex.hh"

#include <string>
#include <type_traits>

#define _unused(x) ((void)(x))
 
namespace XtcData
{

class VarDef;

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
    DescData(NameIndex& nameindex, Xtc& parent, NamesId& namesId) :
        _shapesdata(*new (parent) ShapesData(namesId)),
        _nameindex(nameindex),
        _numarrays(0)
    {
        Names& names = _nameindex.names();
        assert(names.num()<MaxNames);
        _unused(names);
        _offset[0]=0;
        _numentries=0;
    }

    DescData(NameIndex& nameindex, Xtc& parent, VarDef& V, NamesId& namesId) :
        _shapesdata(*new (parent) ShapesData(namesId)),
        _nameindex(nameindex),
        _numarrays(0)
    {
        Names& names = _nameindex.names();
        assert(names.num()<MaxNames);
        _unused(names);
        _offset[0]=0;
        _numentries=0;
    }
    void set_array_shape(unsigned index, unsigned shapeIndex, unsigned shape[MaxRank]) {

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
        DescribedData(Xtc& parent, NameIndex& nameindex, NamesId& namesId) :
            DescData(nameindex, parent, namesId), _parent(parent)
        {
            new (&_shapesdata) Data(_parent);
        }

        DescribedData(Xtc& parent, NamesVec& NamesVec, NamesId& namesId) :
            DescData(NamesVec[namesId.value()], parent, namesId), _parent(parent)
        {
            new (&_shapesdata) Data(_parent);
        }

        void* data() {return _shapesdata.data().payload();}

        void set_data_length(unsigned size) {
            // now that data has arrived manually update with the number of bytes received
            _shapesdata.data().alloc(size, _shapesdata, _parent);
        }

        void set_array_shape(unsigned index, unsigned shape[MaxRank]) {
            if (_numarrays==0) {
                // add the xtc that will hold the shapes of arrays
                Shapes& shapes = *new (&_shapesdata) Shapes(_parent);
                shapes.alloc(_nameindex.shapeMap().size()*sizeof(Shape),
                             _shapesdata, _parent);
            }
            unsigned shapeIndex = _numarrays;
            DescData::set_array_shape(index, shapeIndex, shape);
        }
    private:
        Xtc& _parent;
    };

    class CreateData : public DescData {     
    public:

        CreateData(Xtc& parent, NamesVec& NamesVec, NamesId& namesId) :
            DescData(NamesVec[namesId.value()], parent, namesId), _parent(parent)
        {
            Shapes& shapes = *new (&_shapesdata) Shapes(_parent);
            Names& names = _nameindex.names();
            shapes.alloc(names.numArrays()*sizeof(Shape), _shapesdata, _parent);
            new (&_shapesdata) Data(_parent);
        }

        CreateData(Xtc& parent, NamesVec& NamesVec, VarDef& V, NamesId& namesId) :
            DescData(NamesVec[namesId.value()], parent, V, namesId), _parent(parent)
        {
            Shapes& shapes = *new (&_shapesdata) Shapes(_parent);
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
            CreateData::set_array_shape(index, shape);

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

        void set_array_shape(unsigned index,unsigned shape[MaxRank]) {
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

}; // namespace XtcData

#endif // DESCDATA__H
