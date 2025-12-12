//---------------------------------------------------
// Adopted to lcls2 on 2018-06-06 by Mikhail Dubrovin
//---------------------------------------------------

#include "psalg/calib/ArrayIO.hh"

#include <iostream>  // cout, puts
#include <stdlib.h>  // atoi
#include <cstring>   // memcpy
#include <sstream>   // stringstream, streambuf

using namespace psalg;

namespace psalg {

//-----------------------------

template <typename T>
ArrayIO<T>::ArrayIO(const std::string& fname, void *buf)
  : _ctor(1)
  , _fname(fname)
  , _buf(buf)
{
  _init();
  _load_array();
}

//-----------------------------


template <typename T>
ArrayIO<T>::~ArrayIO()
{
  MSG(TRACE, "DESTRUCTOR for ctor:" << _ctor << " fname=" << _fname);
  //if(_buf_own) delete _buf_own;
}

//-----------------------------

template <typename T>
void ArrayIO<T>::_init()
{
   MSG(TRACE, "ctor:" << _ctor << " fname=" << _fname);
  _status = ArrayIO<T>::UNDEFINED;
}

//-----------------------------

template <typename T>
void ArrayIO<T>::_load_array()
{
    MSG(TRACE, "Load file " << _fname);

    _count_1st_line = 0;
    _count_str_data = 0;
    _count_str_comt = 0;
    _count_data     = 0;

    // open file
    std::ifstream in(_fname.c_str());
    if (in.good()) { MSG(TRACE, "File is open"); }
    else { 
      MSG(WARNING, "Failed to open file: \"" << _fname << "\""); 
	_status = ArrayIO<T>::UNREADABLE; // std::string("file is unreadable");
        return;
    }

    // read and process all strings
    std::string s; 
    while(getline(in,s)) {

        // 1. parse lines with comments marked by # in the 1st position
        if(s[0] == '#') _parse_str_of_comment(s.substr(1));

        // 2. skip empty lines 
        else if (s.empty()) continue; 

        // 3. parse 1st line and load other data
        else _load_data(in,s);
    }

    //close file
    in.close();
    _status = ArrayIO<T>::LOADED; // std::string("loaded from file");

    MSG(TRACE, "Loaded data from file: \"" << _fname << "\""
         << " Input array " << _nda << " of type " << _dtype_name);
}

//-----------------------------

template <typename T>
void ArrayIO<T>::_parse_str_of_comment(const std::string& s)
{
    _count_str_comt ++;
    MSG(TRACE, "_parse_str_of_comment " << s);

    std::stringstream ss(s);
    std::string key;
    std::string value;
    std::string fld;

    ss >> key;
    do {ss >> fld; if(! value.empty()) value+=' '; value+=fld;} while(ss.good()); 
    // MSG(DEBUG, "==== k:" << key << " v:" << value);
    if     (key=="SHAPE")    _parse_shape(value);
    else if(key=="DATATYPE") _dtype_name = value;
    else MSG(DEBUG, "     skip parsing for k:" << key << " v:" << value);
}

//-----------------------------
/// Converts string like "(704,768)" to array _shape={704,768} and counts _ndim
template <typename T>
void ArrayIO<T>::_parse_shape(const std::string& str) {
  std::string s(str.substr(1, str.size()-2)); // remove '(' and ')'
  for (std::string::iterator it = s.begin(); it!=s.end(); ++it) if(*it==',') *it=' '; 
  //MSGLOG(__name__(), DEBUG, "TODO : _parse_shape: " + s + '\n');
  std::stringstream ss(s);
  std::string fld;
  _ndim=0;
  do {ss >> fld; 
    shape_t v = (shape_t)atoi(fld.c_str());
    //cout << "   " << v << endl;
    _shape[_ndim++]=v;
  } while(ss.good());

  _nda.set_shape(_shape, _ndim);
  _size = _nda.size();
}

//-----------------------------

template <typename T>
void ArrayIO<T>::_load_data(std::ifstream& in, const std::string& str)
{
  MSG(TRACE, "TODO : _load_data " << str.substr(0,50));
  //cout << '*';
  
  //if (! _count_str_data++) create_ndarray();

    _count_str_data ++;

    // parse the 1st string
    T val;
    _nda.set_data_buffer(_buf);
    _pdata = _nda.data();

    std::stringstream ss(str);
    while (ss >> val && _count_data < _size) { //&& _count_data != _size) { 
      *_pdata++ = val;
      ++_count_data;
      //cout << ' ' << val;
    }
    //cout << '\n';

    _count_1st_line = _count_data;

    // load all data by the end
    //while (in >> val && _count_data < _size) {
    while (in >> val) {
      *_pdata++ = val;
      ++_count_data;
      //cout << ' ' << val;
    }

    if(_count_data > _size) {
        MSG(WARNING, "Input number of data fields " << _count_data << " exceeds expected array size " << _size);
    } 
    else if (_count_data < _size) {
        MSG(WARNING, "Input number of data fields " << _count_data << " is less then expected array size " << _size);
    }
    else {
        MSG(DEBUG, "counter of data fields = " << _count_data 
                   << " #lines = " << _count_str_data
	           << " 1-st line size = " << _count_1st_line);
    }
}

//-----------------------------

} // namespace psalg

//-----------------------------
//-----------------------------
//-----------------------------

template class psalg::ArrayIO<int>; 
template class psalg::ArrayIO<unsigned>; 
template class psalg::ArrayIO<unsigned short>; 
template class psalg::ArrayIO<float>; 
template class psalg::ArrayIO<double>; 
template class psalg::ArrayIO<int16_t>; 
template class psalg::ArrayIO<uint8_t>; 

//-----------------------------
//-----------------------------
//-----------------------------
