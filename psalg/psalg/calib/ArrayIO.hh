#ifndef PSALG_ARRAYIO_H
#define PSALG_ARRAYIO_H

//---------------------------------------------------
// Adopted to lcls2 on 2018-06-06 by Mikhail Dubrovin
//---------------------------------------------------
/** Usage example
 *
 *  #include "psalg/calib/ArrayIO.hh"
 * ArrayIO<float> aio("/reg/neh/home/dubrovin/LCLS/con-detector/work/nda-xpptut15-r0260-XcsEndstation.0_Epix100a.1-e000030.txt");
 *
 * NDArray<float>& arr = aio.ndarray();
 * std::cout << "ndarray: " << arr);
 */

#include <string>
#include <fstream>   // in *.hh
#include <stdint.h>  // uint8_t, uint32_t, etc.
#include <cstring>   // memcpy

#include "psalg/calib/Types.hh" // shape_t, size_t
#include "psalg/calib/NDArray.hh" // NDArray
#include "psalg/utils/Logger.hh" // for MSG

using namespace std;
using namespace psalg;

namespace psalg {

//-------------------

template <typename T>
class ArrayIO {

public:

  typedef psalg::types::shape_t shape_t; // uint32_t
  typedef psalg::types::size_t  size_t;  // uint32_t

  enum        STATUS     {LOADED=0, DEFAULT,   UNREADABLE,   UNDEFINED};
  std::string STRAUS[4]={"LOADED", "DEFAULT", "UNREADABLE", "UNDEFINED"};

  //ArrayIO();
  ArrayIO(const std::string& fname, void *buf=0);
  ~ArrayIO();

  //inline char* __name__(){return (char*)"ArrayIO";}
  inline const STATUS status() const {return _status;}
  inline const std::string str_status() const {return STRAUS[_status];}
  inline const std::string& dtype_name() const {return _dtype_name;}

  NDArray<T>& ndarray(){return _nda;};

private:

  //Stack* _stack; // reserve memory for data

  unsigned    _ctor;
  std::string _fname;
  void*       _buf;
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

  NDArray<T>  _nda;

  void _init();

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
