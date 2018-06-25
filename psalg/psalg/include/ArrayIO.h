
#ifndef PSALG_ARRAYIO_H
#define PSALG_ARRAYIO_H

//---------------------------------------------------
// Adopted to lcls2 on 2018-06-06 by Mikhail Dubrovin
//---------------------------------------------------

#include <map>
#include <string>
#include <iostream> // for cout, puts etc.
#include <fstream>
#include <stdint.h>  // uint8_t, uint32_t, etc.

#include "psalg/include/ArrayMetadata.h"
#include "xtcdata/xtc/Array.hh" // for Array

#include "psalg/include/Logger.h" // for MsgLog
//#include <vector>
//#include "pdscalibdata/GlobalMethods.h"

using namespace std;

namespace psalg {

//-------------------

template <typename TDATA>
class ArrayIO {

public:

  typedef TDATA data_t;
  typedef ArrayMetadata::shape_t shape_t; // uint32_t
  typedef ArrayMetadata::size_t size_t; // uint32_t

  enum        STATUS     {LOADED=0, DEFAULT,   UNREADABLE,   UNDEFINED};
  std::string STRAUS[4]={"LOADED", "DEFAULT", "UNREADABLE", "UNDEFINED"};

  ArrayIO(const std::string& fname);

  ~ArrayIO();

  inline char* __name__(){return (char*)"ArrayIO";}
  inline const STATUS status() const {return _status;}
  inline const std::string& dtype_name() const {return _dtype_name;};

protected:

private:

  //Array<TDATA>* p_nda;

  std::string _fname;
  unsigned    _ctor;
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

/*
template <typename TDATA>
class ArrayIO {

  //static const unsigned c_ndim = NDIM;

public:

  ArrayIO(const std::string& fname
	   ,const shape_t* shape_def
	   ,const TDATA& val_def=TDATA(0) 
	   ,const unsigned print_bits=0377);

  ArrayIO(const std::string& fname
	   ,const Array<const TDATA>& nda_def
	   ,const unsigned print_bits=0377);

  /// Returns number of dimensions of array.
  unsigned int ndim() const {return NDIM;}

  /// Access methods
  /// prints recognized templated parameters.
  void print();

  /// Prints input file line-by-line.
  void print_file();

  /// Loads (if necessary) array from file and print it.
  void print_array();

  /// Loads (if necessary) array from file and returns it.
  Array<TDATA, NDIM>& get_array(const std::string& fname = std::string());
  //Array<const TDATA, NDIM>& get_array(const std::string& fname = std::string());

  /// Returns string with status of calibration constants.
  std::string str_status();

  /// Returns enumerated status of calibration constants.
  STATUS status(){return m_status;}

  /// Returns string with info about array.
  std::string str_array_info();

  /// Returns string of shape.
  std::string str_shape();

  /// Static method to save array in file with internal metadata and external comments
  static void save_ndarray(const Array<const TDATA, NDIM>& nda, 
                           const std::string& fname,
                           const std::vector<std::string>& vcoms = std::vector<std::string>(), 
	                   const unsigned& print_bits=0377);

protected:

private:

  /// Data members  

  Array<TDATA, NDIM>* p_nda;

  std::string m_fname;
  TDATA       m_val_def;
  const Array<const TDATA, NDIM> m_nda_def;
  Array<TDATA, NDIM> m_nda_empty;
  unsigned    m_print_bits;

  unsigned    m_ndim;
  size_t      m_size;
  shape_t     m_shape[NDIM];
  std::string m_str_type;
  DATA_TYPE   m_enum_type;

  TDATA* p_data;

  /// true if the file name non empty anf file is readable
  bool file_is_available();

  /// creates Array<TYPE,NDIM> with shape from constructor parameter or metadata.
  void create_array(const bool& fill_def=false);

};

*/

} // namespace psalg

#endif // PSALG_ARRAYIO_H
