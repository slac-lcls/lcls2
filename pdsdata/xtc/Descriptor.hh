#ifndef DESCRIPTOR__H
#define DESCRIPTOR__H

#include <cstring>
#include <iostream>
#include <array>
#include <cassert>
#include <chrono>
#include <vector>
#include <type_traits>

enum Type
{
    INT,
    FLOAT,
    UINT16_ARRAY,
    FLOAT_ARRAY,
};

template<typename T>
struct Array
{
    Array(uint8_t* buffer)
    {
        data = reinterpret_cast<T*>(buffer);
    }
    T& operator() (int i, int j)
    {
        return data[i*shape[1] + j];
    }
    const T& operator() (int i, int j) const
    {
        return data[i*shape[1] + j];
    }
    T& operator() (int i, int j, int k)
    {
        return data[(i*shape[1] + j)*shape[2] + k];
    }
    const T& operator() (int i, int j, int k) const
    {
        return data[(i*shape[1] + j)*shape[2] + k];
    }

    T* data;
    std::vector<int> shape;
};

template<typename>
struct is_vec : std::false_type {};

template<typename T>
struct is_vec<Array<T>> : std::true_type {};

struct Field
{
    char name[256];
    Type type;
    int offset;
    int rank;
    std::array<int, 5> shape;
};

class Descriptor
{
public:
    Descriptor(uint8_t* buffer);

    inline Field* get_field_by_index(int index)
    {
        return reinterpret_cast<Field*>(_buffer + sizeof(int) + index*sizeof(Field));
    }

    Field* get_field_by_name(const char* name);

private:
    uint8_t* _buffer;
    int _num_fields;
};

// all fundamental types
template<typename T>
typename std::enable_if<std::is_fundamental<T>::value, T>::type
get_value(const Field& field, uint8_t* buffer) {
    return *reinterpret_cast<T*>(buffer + field.offset);
}

#endif // DESCRIPTOR__H
