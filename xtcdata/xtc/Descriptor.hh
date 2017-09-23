#ifndef DESCRIPTOR__H
#define DESCRIPTOR__H

#include <cstring>
#include <iostream>
#include <type_traits>
#include <vector>
#include <assert.h>

#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/TypeId.hh"

enum DataType { UINT8, UINT16, INT32, FLOAT, DOUBLE };

static int get_element_size(DataType type)
{
    const static int element_sizes[] = { sizeof(uint8_t), sizeof(uint16_t), sizeof(int32_t),
                                         sizeof(float), sizeof(double) };
    return element_sizes[type];
}

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

class Field
{
public:
    static const int maxNameSize = 256;
    Field(const char* tmpname, DataType tmptype, int tmpoffset)
    {
        strncpy(name, tmpname, maxNameSize);
        type = tmptype;
        offset = tmpoffset;
        rank = 0;
    }

    Field(const char* tmpname, DataType tmptype, int tmpoffset, int tmprank, int tmpshape[5])
    {
        strncpy(name, tmpname, maxNameSize);
        type = tmptype;
        offset = tmpoffset;
        rank = tmprank;
        memcpy(shape, tmpshape, sizeof(int) * rank);
    }
    char name[maxNameSize];
    DataType type;
    uint32_t offset;
    uint32_t rank;
    uint32_t shape[5]; // in an ideal world this would have length "rank"
};

// this class updates the "parent" DescData Xtc extent at the same time
// the child Desc or Data extent is increased.  the hope is that this
// will make management of the DataDesc less error prone, at the price
// of some performance (more calls to DescData::alloc())
class DescDataManager : public XtcData::Xtc
{
public:
    DescDataManager(XtcData::TypeId typeId) : XtcData::Xtc(typeId)
    {
    }
    void* alloc(uint32_t size, XtcData::Xtc& descdata) {
        descdata.alloc(size);
        return XtcData::Xtc::alloc(size);
    }
};

class Data : public DescDataManager
{
public:
    Data() : DescDataManager(XtcData::TypeId(XtcData::TypeId::Data,0)) {}
};

class Desc : public DescDataManager
{
public:
    Desc() : DescDataManager(XtcData::TypeId(XtcData::TypeId::Desc,0))
    {
    }
    Field* get_field_by_name(const char* name);

    Field& get(int index)
    {
        Field& tmpptr = ((Field*)(this + 1))[index];
        return tmpptr;
    }

    // Add new scalar to Desc
    void add(const char* name, DataType type, uint32_t& offset, Xtc& descdata)
    {
        if (num_fields() == 0) offset = 0;
        new (&get(num_fields())) Field(name, type, offset);
        offset += get_element_size(type);
        alloc(sizeof(Field), descdata);
    }

    // Add new array to Desc
    void add(const char* name, DataType type, int rank, int shape[5], uint32_t& offset, XtcData::Xtc& descdata)
    {
        if (num_fields() == 0) offset = 0;
        new (&get(num_fields())) Field(name, type, offset, rank, shape);
        int num_elements = 1;
        for (int i = 0; i < rank; i++) {
            num_elements *= shape[i];
        }
        offset += num_elements * get_element_size(type);
        alloc(sizeof(Field), descdata);
    }

    uint32_t num_fields()
    {
        return sizeofPayload() / sizeof(Field);
    }

private:
};

// generic data class that holds the size of the variable-length data and
// the location of the descriptor
class DescData : public XtcData::Xtc
{
public:
    DescData() : XtcData::Xtc(XtcData::TypeId(XtcData::TypeId::DescData,0)) {}

    Data& data()
    {
        if (_firstIsDesc()) {
            Data& d = reinterpret_cast<Data&>(_second());
            assert(d.contains.id()==XtcData::TypeId::Data);
            return d;
        }
        else {
            Data& d = reinterpret_cast<Data&>(_first());
            assert(d.contains.id()==XtcData::TypeId::Data);
            return d;
        }
    }

    Desc& desc()
    {
        if (_firstIsDesc()) {
            Desc& d = reinterpret_cast<Desc&>(_first());
            assert(d.contains.id()==XtcData::TypeId::Desc);
            return d;
        }
        else {
            Desc& d = reinterpret_cast<Desc&>(_second());
            assert(d.contains.id()==XtcData::TypeId::Desc);
            return d;
        }
    }

    // all fundamental types
    template <typename T>
    typename std::enable_if<std::is_fundamental<T>::value, T>::type get_value(const char* name)
    {
        Field* field = desc().get_field_by_name(name);
        return *reinterpret_cast<T*>(data().payload() + field->offset);
    }

    template <typename T>
    typename std::enable_if<std::is_fundamental<T>::value, T>::type set_value(const char* name, T val)
    {
        Field* field = desc().get_field_by_name(name);
        T* ptr = reinterpret_cast<T*>(data().payload() + field->offset);
        *ptr = val;
        data().alloc(sizeof(T), *this);
    }

    // for all array types
    template <typename T>
    typename std::enable_if<is_vec<T>::value, T>::type get_value(const char* name)
    {
        Field* field = desc().get_field_by_name(name);
        T array(data().payload() + field->offset);
        array._shape.resize(field->rank);
        for (int i = 0; i < field->rank; i++) {
            array._shape[i] = field->shape[i];
        }
        return array;
    }

private:
    XtcData::Xtc& _first() {
        return *(XtcData::Xtc*)payload();
    }

    XtcData::Xtc& _second() {
        return *_first().next();
    }

    bool _firstIsDesc() {
        return _first().contains.id()==XtcData::TypeId::Desc;
    }

};

#endif // DESCRIPTOR__H
