#ifndef DESCRIPTOR__H
#define DESCRIPTOR__H

#include <array>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <vector>

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
    uint32_t shape[5];
};

class DescData;

class Info
{
public:
    enum Type { DataFirst = 0, DescFirst = 0x80000000 };
    Info(Type type) : _sizeofPayloadAndType(type)
    {
    }
    uint8_t* next()
    {
        return (uint8_t*)(this) + size();
    }
    void extend(unsigned bytes)
    {
        _sizeofPayloadAndType += bytes;
    }
    uint8_t* payload()
    {
        return (uint8_t*)(this + 1);
    }
    uint32_t size()
    {
        return sizeofPayload() + sizeof(Info);
    }

protected:
    friend class DescData;
    uint32_t sizeofPayload()
    {
        return _sizeofPayloadAndType & (~DescFirst);
    }
    bool isDesc()
    {
        return _sizeofPayloadAndType & DescFirst;
    }
    uint32_t _sizeofPayloadAndType;
};

class Descriptor : public Info
{
public:
    Descriptor() : Info(DescFirst)
    {
    }
    Field* get_field_by_name(const char* name);

    Field& get(int index)
    {
        Field& tmpptr = ((Field*)(this + 1))[index];
        return tmpptr;
    }

    // Add new scalar to Descriptor
    void add(const char* name, DataType type, uint32_t& offset)
    {
        if (num_fields() == 0) offset = 0;
        new (&get(num_fields())) Field(name, type, offset);
        offset += get_element_size(type);
        _add_field();
    }

    // Add new array to Descriptor
    void add(const char* name, DataType type, int rank, int shape[5], uint32_t& offset)
    {
        if (num_fields() == 0) offset = 0;
        new (&get(num_fields())) Field(name, type, offset, rank, shape);
        int num_elements = 1;
        for (int i = 0; i < rank; i++) {
            num_elements *= shape[i];
        }
        offset += num_elements * get_element_size(type);
        _add_field();
    }

    uint32_t num_fields()
    {
        return sizeofPayload() / sizeof(Field);
    }

private:
    void _add_field()
    {
        extend(sizeof(Field));
    }
};

class Data : public Info
{
public:
    Data() : Info(DataFirst)
    {
    }
};

// generic data class that holds the size of the variable-length data and
// the location of the descriptor
class DescData
{
public:
    // constructor for use by FEX software
    DescData()
    {
        new (this) Descriptor();
    }

    uint8_t* data()
    {
        return _data().payload();
    }

    Descriptor& desc()
    {
        if (_first().isDesc())
            return reinterpret_cast<Descriptor&>(_first());
        else
            return reinterpret_cast<Descriptor&>(_second());
    }

    // all fundamental types
    template <typename T>
    typename std::enable_if<std::is_fundamental<T>::value, T>::type get_value(const char* name)
    {
        Field* field = desc().get_field_by_name(name);
        return *reinterpret_cast<T*>(data() + field->offset);
    }

    template <typename T>
    typename std::enable_if<std::is_fundamental<T>::value, T>::type set_value(const char* name, T val)
    {
        Field* field = desc().get_field_by_name(name);
        T* ptr = reinterpret_cast<T*>(data() + field->offset);
        *ptr = val;
        _data().extend(sizeof(T));
    }

    // for all array types
    template <typename T>
    typename std::enable_if<is_vec<T>::value, T>::type get_value(const char* name)
    {
        Field* field = desc().get_field_by_name(name);
        T array(data() + field->offset);
        array._shape.resize(field->rank);
        for (int i = 0; i < field->rank; i++) {
            array._shape[i] = field->shape[i];
        }
        return array;
    }

private:
    Data& _data()
    {
        if (_first().isDesc())
            return reinterpret_cast<Data&>(_second());
        else
            return reinterpret_cast<Data&>(_first());
    }

    Info& _first()
    {
        return *reinterpret_cast<Info*>(this);
    }

    Info& _second()
    {
        return *reinterpret_cast<Info*>(reinterpret_cast<uint8_t*>(this) + _first().size());
    }
};

#endif // DESCRIPTOR__H
