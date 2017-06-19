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
    FLOAT_ARRAY,
};

int get_element_size(Type& type)
{
    const static int element_sizes[] = 
    {
        4, // INT
        4, // FLOAT
        4  // FLOAT_ARRAY
    };
    return element_sizes[type];
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
  static const int maxNameSize=256;
  Field(const char* tmpname, Type tmptype, int tmpoffset) {
    strncpy(name, tmpname, maxNameSize);
    type = tmptype;
    offset = tmpoffset;
    rank = 1;
    shape[0] = 1;
    shape[1] = 0;
    shape[2] = 0;
    shape[3] = 0;
    shape[4] = 0;
  }
    char name[maxNameSize];
    Type type;
    int offset;
    int rank;
    int shape[5];
};

class Descriptor
{
public:
  Descriptor();

    // inline Field* get_field_by_index(int index)
    // {
    //     return reinterpret_cast<Field*>(_buffer + sizeof(int) + index*sizeof(Field));
    // }

    // Field* get_field_by_name(const char* name);

  Field& get(int index) {
    Field& tmpptr = ((Field*)(this+1))[index];
    return tmpptr;
  }

  int num_fields;
};

class DescriptorManager {
public:
  DescriptorManager(void *ptrToDesc) : _desc(*new(ptrToDesc) Descriptor) {
    _offset = 0;
  }

  void add(const char* name, Type type) {
    new(&_desc.get(_desc.num_fields)) Field(name, type, _offset);
    _offset += get_element_size(type);
    _desc.num_fields++;
  }

  void add(const char* name, Type type, int rank, int shape[5]) {
      
      int num_elements = 1;
      for (int i=0; i<5; i++) {
          num_elements *= shape[i];
    }
     _offset += num_elements * get_element_size(type);
    _desc.num_fields++;
  }

  int size() {
    return sizeof(Descriptor)+_desc.num_fields*sizeof(Field);
  }
  Descriptor &_desc;
private:
  int _offset;
};

// all fundamental types
template<typename T>
typename std::enable_if<std::is_fundamental<T>::value, T>::type
get_value(const Field& field, uint8_t* buffer) {
    return *reinterpret_cast<T*>(buffer + field.offset);
}

#endif // DESCRIPTOR__H
