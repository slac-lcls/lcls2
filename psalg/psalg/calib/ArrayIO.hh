/**
 * @file
 * @brief Declares psalg::ArrayIO, which loads an NDArray from a text file with '#' metadata lines.
 */
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

/**
 * Loads an array of T from a text file in the constructor. Lines starting with '#' are metadata ("SHAPE (d0,d1,...)" sets the shape, "DATATYPE x" the type name); the first other non-empty line and everything after it are read as whitespace-separated values.
 * Instantiated in ArrayIO.cc for int, unsigned, unsigned short and float.
 */
template <typename T>
class ArrayIO {

public:

  /** Alias for psalg::types::shape_t (uint32_t). */
  typedef psalg::types::shape_t shape_t; // uint32_t
  /** Alias for psalg::types::size_t (uint32_t). */
  typedef psalg::types::size_t  size_t;  // uint32_t

  /** Load status reported by status(). */
  enum        STATUS     {LOADED=0, /**< 0: the file was read (also set when the value count differs from the shape size). */ DEFAULT, /**< 1: not set anywhere in ArrayIO.cc. */   UNREADABLE, /**< 2: the file could not be opened. */   UNDEFINED /**< 3: initial value before loading. */ };
  std::string STRAUS[4]={"LOADED", "DEFAULT", "UNREADABLE", "UNDEFINED"};  ///< Names of the STATUS values, used by str_status().

  //ArrayIO();
  /**
   * Read fname right away. Values go into buf if it is non-null, otherwise into a buffer the NDArray allocates.
   * Logs a warning if the file cannot be opened (status UNREADABLE) or if the number of values differs from the shape size; extra values are still written past the end of the buffer.
   */
  ArrayIO(const std::string& fname, void *buf=0);
  /** Destructor; only logs a trace message. */
  ~ArrayIO();

  //inline char* __name__(){return (char*)"ArrayIO";}
  /** Return the load status. */
  inline const STATUS status() const {return _status;}
  /** Return the status name ("LOADED", "DEFAULT", "UNREADABLE" or "UNDEFINED"). */
  inline const std::string str_status() const {return STRAUS[_status];}
  /** Return the DATATYPE value read from the file (empty if there was none). */
  inline const std::string& dtype_name() const {return _dtype_name;}

  /** Return the loaded array. */
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
