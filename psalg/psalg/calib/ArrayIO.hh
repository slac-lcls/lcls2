
#ifndef PSALG_ARRAYIO_H
#define PSALG_ARRAYIO_H

//---------------------------------------------------
// Adopted to lcls2 on 2018-06-06 by Mikhail Dubrovin
//---------------------------------------------------

//#include <map>
//#include <vector>
//#include <iostream> // cout, puts etc.
#include <string>
#include <fstream> // in *.hh
#include <stdint.h>  // uint8_t, uint32_t, etc.

//#include "xtcdata/xtc/Array.hh" // Array
#include "psalg/alloc/AllocArray.hh"
#include "psalg/calib/ArrayMetadata.hh"
#include "psalg/utils/Logger.hh" // for MSG

using namespace std;
using namespace psalg;

namespace psalg {

//-------------------

template <typename T>
class ArrayIO {

public:

  typedef T data_t;
  typedef ArrayMetadata::shape_t shape_t; // uint32_t
  typedef ArrayMetadata::size_t size_t; // uint32_t

  //enum {MAXBUFSIZE=704*768};
  enum        STATUS     {LOADED=0, DEFAULT,   UNREADABLE,   UNDEFINED};
  std::string STRAUS[4]={"LOADED", "DEFAULT", "UNREADABLE", "UNDEFINED"};

  ArrayIO(const std::string& fname, void *buf=0);

  ~ArrayIO();

  inline char* __name__(){return (char*)"ArrayIO";}
  inline const STATUS status() const {return _status;}
  inline const std::string str_status() const {return STRAUS[_status];}
  inline const std::string& dtype_name() const {return _dtype_name;}

protected:

private:

  //AllocArray<T> _arr;
  //Array<T>* _nda;
  //AllocArray1D<T> _1da; // reserve memory for data
  //AllocArray1D<float> _1da; // reserve memory for data

  Stack* _stack; // reserve memory for data

  unsigned    _ctor;
  std::string _fname;
  T*          _buf_ext;
  T*          _buf_own;
  T*          _pdata;

  STATUS      _status;

  size_t      _count_1st_line;
  size_t      _count_str_data;
  size_t      _count_str_comt;
  size_t      _count_data;

  shape_t     _shape[10];
  size_t      _ndim;
  size_t      _size;
  std::string _dtype_name;

  ArrayMetadata _metad;

  //std::map<std::string,std::string> _metad;

  void _init();

  void _reset_data_pointer();

  /// loads metadata and data from file
  void _load_array();

  /// parser for comment lines and metadata from file with array
  void _parse_str_of_comment(const std::string& str);

  /// parser for comment lines and metadata from file with array
  void _parse_shape(const std::string& str);

  /// creates array, begins to fill data from the 1st string and reads data by the end
  void _load_data(std::ifstream& in, const std::string& str);

  /// Copy constructor and assignment are disabled by default
  ArrayIO(const ArrayIO&) ;
  ArrayIO& operator = (const ArrayIO&) ;
};

} // namespace psalg

#endif // PSALG_ARRAYIO_H
