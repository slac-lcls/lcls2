#ifndef DESCRIPTOR__H
#define DESCRIPTOR__H

#include <cstring>
#include <array>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <vector>

enum Type {
  UINT8,
  UINT16,
  INT32,
  FLOAT,
  DOUBLE
};

int get_element_size(Type& type);

template <typename T> struct Array {
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

template <typename> struct is_vec : std::false_type {
};

template <typename T> struct is_vec<Array<T>> : std::true_type {
};

struct Field {
  static const int maxNameSize = 256;
  Field(const char* tmpname, Type tmptype, int tmpoffset)
  {
    strncpy(name, tmpname, maxNameSize);
    type = tmptype;
    offset = tmpoffset;
    rank = 0;
  }

  Field(const char* tmpname, Type tmptype, int tmpoffset, int tmprank, int tmpshape[5])
  {
    strncpy(name, tmpname, maxNameSize);
    type = tmptype;
    offset = tmpoffset;
    rank = tmprank;
    memcpy(shape, tmpshape, sizeof(int)*rank);
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
  Field* get_field_by_name(const char* name);

  Field& get(int index)
  {
    Field& tmpptr = ((Field*)(this + 1))[index];
    return tmpptr;
  }

  int num_fields;
};

class DescriptorManager
{
public:
  DescriptorManager(void* ptrToDesc) : _desc(*new(ptrToDesc) Descriptor)
  {
    _offset = 0;
  }

  // Add new scalar to Descriptor
  void add(const char* name, Type type)
  {
    new(&_desc.get(_desc.num_fields)) Field(name, type, _offset);
    _offset += get_element_size(type);
    _desc.num_fields++;
  }

  // Add new array to Descriptor
  void add(const char* name, Type type, int rank, int shape[5])
  {
    new(&_desc.get(_desc.num_fields)) Field(name, type, _offset, rank, shape);
    int num_elements = 1;
    for(int i = 0; i < rank; i++) {
      num_elements *= shape[i];
    }
    _offset += num_elements * get_element_size(type);
    _desc.num_fields++;
  }

  int size()
  {
    return sizeof(Descriptor) + _desc.num_fields * sizeof(Field);
  }
  Descriptor& _desc;

private:
  int _offset;
};

// generic data class that holds the size of the variable-length data and
// the location of the descriptor
class Data
{
public:
  enum Type {DataFirst, DescFirst=0x80000000};

private:
  class Info {
  public:
    Info(uint32_t size, Type type) : _sizeOfFirstAndType(size | type) {}
    uint32_t sizeofFirst() {return _sizeOfFirstAndType&(~DescFirst);}
    bool     descFirst()   {return _sizeOfFirstAndType&DescFirst;}
  private:
    uint32_t _sizeOfFirstAndType;
  } _info;

  char* _second()
  {
    return reinterpret_cast<char*>(this) + _info.sizeofFirst();
  }

  char* _first()
  {
    return reinterpret_cast<char*>((this + 1));
  }

public:
  // use when data comes from PGP
  Data(uint32_t size) : _info(size,DataFirst) {}
  // use for feature extraction
  Data()              : _info(0,DescFirst) {}

  uint8_t* get_buffer()
  {
    if (_info.descFirst())
      return reinterpret_cast<uint8_t*>(_second());
    else
      return reinterpret_cast<uint8_t*>(_first());
  }

  Descriptor& desc() {
    if (_info.descFirst())
      return *(Descriptor*)_first();
    else
      return *(Descriptor*)_second();
  }

  // all fundamental types
  template <typename T>
  typename std::enable_if<std::is_fundamental<T>::value, T>::type get_value(const char* name)
  {
    Field* field = desc().get_field_by_name(name);
    return *reinterpret_cast<T*>((uint8_t*)(this + 1) + field->offset);
  }

  template <typename T>
  typename std::enable_if<std::is_fundamental<T>::value, T>::type set_value(const char* name, T val)
  {
    Field* field = desc().get_field_by_name(name);
    T* ptr = reinterpret_cast<T*>((uint8_t*)(this + 1) + field->offset);
    *ptr = val;
  }    

  // for all array types
  template <typename T>
  typename std::enable_if<is_vec<T>::value, T>::type get_value(const char* name)
  {
    Field* field = desc().get_field_by_name(name);
    T array((uint8_t*)(this + 1) + field->offset);
    array._shape.resize(field->rank);
    for(int i = 0; i < field->rank; i++) {
      array._shape[i] = field->shape[i];
    }
    return array;
  }

};

#endif // DESCRIPTOR__H
