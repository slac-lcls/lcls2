//---------------------------------------------------
// Adopted to lcls2 on 2018-06-06 by Mikhail Dubrovin
//---------------------------------------------------

#include "psalg/include/ArrayIO.h"
#include "psalg/include/Logger.h" // MsgLog

//#include <algorithm>
//#include <stdexcept>

#include <fstream>
#include <sstream>   // stringstream
#include <stdlib.h>  // atoi
#include <cstring>   // memcpy
#include <sstream>   // stringstream, streambuf

using namespace psalg;

namespace psalg {

//-----------------------------

template <typename TDATA>
ArrayIO<TDATA>::ArrayIO(const std::string& fname)
  : _fname(fname)
  , _ctor(0)
  //  : _nda(0)
  //  , _val_def(0)
  //  , _nda_def(ndarray<const TDATA>())
{
  _init();
  //_size = 0; // actual size is loaded from metadata
  _load_array();
}

//-----------------------------

template <typename TDATA>
ArrayIO<TDATA>::~ArrayIO()
{
  std::stringstream ss;
  MsgLog(__name__(), DEBUG, ss << "DESTRUCTOR for ctor:" << _ctor << " fname=" << _fname);

  //delete p_nda;
}

//-----------------------------

template <typename TDATA>
void ArrayIO<TDATA>::_init()
{
  //std::stringstream ss;
  //MsgLog(__name__(), DEBUG, ss << "ctor:" << _ctor << " fname=" << _fname);
  MSGLOG(__name__(), DEBUG, "ctor:" << _ctor << " fname=" << _fname);
  _status = ArrayIO<TDATA>::UNDEFINED;
  //_metad.clear();
}













//-----------------------------

template <typename TDATA>
void ArrayIO<TDATA>::_load_array()
{
  //// if file is not available - create default ndarray
    //if ((!file_is_available()) && m_ctor>0) { 
    //    if( m_print_bits & 4 ) MsgLog(__name__(), warning, "Use default calibration parameters.");
    //    create_ndarray(true);
    //    m_status = ArrayIO<TDATA>::DEFAULT; // std::string("used default");
    //    return; 
    //}

    //std::stringstream ss;
    MSGLOG(__name__(), DEBUG, "Load file " << _fname << '\n');

    _count_1st_line = 0;
    _count_str_data = 0;
    _count_str_comt = 0;
    _count_data     = 0;

    // open file
    std::ifstream in(_fname.c_str());
    if (in.good()) { 
        MsgLog(__name__(), DEBUG, "File is open");
    }
    else { 
        MsgLog(__name__(), WARNING, "Failed to open file: \"" + _fname + "\""); 
	_status = ArrayIO<TDATA>::UNREADABLE; // std::string("file is unreadable");
        return;
    }

    // read and process all strings
    std::string s; 
    while(getline(in,s)) {
        // cout << str << '\n';

        // 1. parse lines with comments marked by # in the 1st position
        if(s[0] == '#') _parse_str_of_comment(s.substr(1));

        // 2. skip empty lines 
        //else if (s.find_first_not_of(" ")==string::npos) continue; 
        else if (s.empty()) continue; 

        // 3. parse 1st line and load other data
        else _load_data(in,s);
        //else {_load_data(in,s); break;}
    }

    //close file
    in.close();
    _status = ArrayIO<TDATA>::LOADED; // std::string("loaded from file");

    cout << endl;

    //ArrayMetadata amd(_shape, _ndim);
    //std::stringstream msg;
    //MsgLog(__name__(), INFO, msg << "Input array " << amd << " of type " << _dtype_name);

    std::stringstream msg;
    MsgLog(__name__(), INFO, msg << "Input array " << _metad << " of type " << _dtype_name);
}

//-----------------------------

template <typename TDATA>
void ArrayIO<TDATA>::_parse_str_of_comment(const std::string& s)
{
    _count_str_comt ++;
    // cout << "comment, str.size()=" << str.size() << '\n';
    // cout << "TO-DO parse cmt: " << s << '\n';
    std::stringstream msg;
    MsgLog(__name__(), DEBUG, msg << "TODO : _parse_str_of_comment " << s);

    std::stringstream ss(s);
    std::string key;
    std::string value;
    std::string fld;
    //std::vector<std::string> fields;

    ss >> key;
    do {ss >> fld; if(! value.empty()) value+=' '; value+=fld;} while(ss.good()); 
    cout << " k:" << key << " v:" << value << endl;
    //_metad[key] = value;

    if     (key=="SHAPE")    _parse_shape(value);
    else if(key=="DATATYPE") _dtype_name = value;
}

//-----------------------------
/// Converts string like "(704,768)" to array _shape={704,768} and counts _ndim
template <typename TDATA>
void ArrayIO<TDATA>::_parse_shape(const std::string& str) {
  std::string s(str.substr(1, str.size()-2)); // remove '(' and ')'
  for (std::string::iterator it = s.begin(); it!=s.end(); ++it) if(*it==',') *it=' '; 
  //MsgLog(__name__(), DEBUG, "TODO : _parse_shape: " + s + '\n');
  std::stringstream ss(s);
  std::string fld;
  _ndim=0;
  do {ss >> fld; 
    shape_t v = (shape_t)atoi(fld.c_str());
    cout << "   " << v << endl;
    _shape[_ndim++]=v;
  } while(ss.good());

  _metad.set(_shape, _ndim);
  _size = _metad.size();

}

//-----------------------------

template <typename TDATA>
void ArrayIO<TDATA>::_load_data(std::ifstream& in, const std::string& str)
{
  //std::stringstream ss;
  //MsgLog(__name__(), DEBUG, ss << "TODO : _load_data " << str.substr(0,50));
  //cout << '*';
  
  //if (! _count_str_data++) create_ndarray();

    _count_str_data ++;

    // parse the 1st string
    TDATA val;
    //TDATA* it=p_data; 

    std::stringstream ss(str);
    while (ss >> val && _count_data < _size) { //&& _count_data != _size) { 
      // *it++ = val;
      ++_count_data;
      cout << ' ' << val;
    }

    _count_1st_line = _count_data;

    // load all data by the end
    while(in >> val && _count_data < _size) {
      //*it++ = val;
      ++_count_data;
      //if(!(_count_data < _size)) {
      //	MsgLog(__name__(), WARNING, "Input number of data fields exceed expected array size");
      //  	break;
      //}
      //cout << ' ' << val;
    }

    cout << "\ncounter of data fields = " << _count_data 
         << " #lines = " << _count_str_data
         << " 1-st line = " << _count_1st_line << '\n';
}


    /*

    // check that we read whole array
    if (m_count_data != m_size) {
      std::stringstream ss;
      ss << "NDArray file:\n  " << m_fname << "\n  does not have enough data: "
         << "read " << m_count_data << " numbers, expecting " << m_size;
      // MsgLog(__name__(), warning, ss.str());
      if( ndim()>1 ) throw std::runtime_error(ss.str());
    }

    // and no data left after we finished reading
    if ( in >> val ) {
      ++ m_count_data;
      std::stringstream ss;
      ss << "NDArray file:\n  " << m_fname << "\n  has extra data: "
         << "read " << m_count_data << " numbers, expecting " << m_size; 
      MsgLog(__name__(), warning, ss.str());
      if( ndim()>1 ) throw std::runtime_error(ss.str());
    }
}

    */






/*
//-----------------------------
//-----------------------------
//-----------------------------
//-----------------------------

template <typename TDATA, unsigned NDIM>
std::string ArrayIO<TDATA>::str_status()
{
  if      (_status == ArrayIO<TDATA>::LOADED)     return std::string("loaded from file");
  else if (_status == ArrayIO<TDATA>::DEFAULT)    return std::string("used default");
  else if (_status == ArrayIO<TDATA>::UNREADABLE) return std::string("file is unreadable");
  else if (_status == ArrayIO<TDATA>::UNDEFINED)  return std::string("undefined...");
  else                                              return std::string("unknown...");
}

//-----------------------------

template <typename TDATA, unsigned NDIM>
bool ArrayIO<TDATA>::file_is_available()
{
  if(m_fname.empty()) {
    if( m_print_bits & 4 ) MsgLog(__name__(), warning, "File name IS EMPTY!");
    return false;
  }

  std::ifstream file(m_fname.c_str());
  if(!file.good()) {
    if( m_print_bits & 8 ) MsgLog(__name__(), warning, "File: " << m_fname << " DOES NOT EXIST!");
    return false;
  }
  file.close();
  return true;  
}

//-----------------------------

template <typename TDATA, unsigned NDIM>
void ArrayIO<TDATA, NDIM>::create_ndarray(const bool& fill_def)
{
    if (p_nda) delete p_nda; // prevent memory leak
    // shape should already be available for
    // m_ctor = 0 - from parsing input file header,
    // m_ctor = 1,2 - from input pars
    p_nda = new ndarray<TDATA, NDIM>(m_shape);
    p_data = p_nda->data();
    m_size = p_nda->size();

    if (m_ctor>0 && fill_def) {
      if      (m_ctor==2) std::memcpy (p_data, m_nda_def.data(), m_size*sizeof(TDATA));
      else if (m_ctor==1) std::fill_n (p_data, m_size, m_val_def);
      else return; // There is no default initialization for ctor=0 w/o shape
    }    
    if( m_print_bits & 16 ) MsgLog(__name__(), info, "Created ndarray of the shape=(" << str_shape() << ")");
    if( m_print_bits & 32 ) MsgLog(__name__(), info, "Created ndarray: " << *p_nda);
}


//-----------------------------

template <typename TDATA, unsigned NDIM>
//ndarray<const TDATA, NDIM>& 
ndarray<TDATA, NDIM>& 
ArrayIO<TDATA, NDIM>::get_ndarray(const std::string& fname)
{
  if ( (!fname.empty()) && fname != m_fname) {
    m_fname = fname;
    load_ndarray(); 
  }

  if (!p_nda) load_ndarray();
  if (!p_nda) {
    if(m_print_bits) MsgLog(__name__(), error, "ndarray IS NOT LOADED! Check file: \"" << m_fname <<"\"");
    return m_nda_empty;
  }
  //std::cout << "TEST in get_ndarray(...):";
  //if (p_nda) std::cout << *p_nda << '\n';

  return *p_nda;
}

//-----------------------------

template <typename TDATA, unsigned NDIM>
void ArrayIO<TDATA, NDIM>::print()
{
    std::stringstream ss; 
    ss << "print()"
       << "\n  Constructor          # " << m_ctor
       << "\n  Number of dimensions : " << ndim()
       << "\n  Data type and size   : " << strOfDataTypeAndSize<TDATA>()
       << "\n  Enumerated data type : " << enumDataType<TDATA>()
       << "\n  String data type     : " << strDataType<TDATA>()
       << '\n';
    MsgLog(__name__(), info, ss.str());
}

//-----------------------------

template <typename TDATA, unsigned NDIM>
void ArrayIO<TDATA, NDIM>::print_file()
{
    if (! file_is_available() ) {
      MsgLog(__name__(), warning, "print_file() : file " << m_fname << " is not available!");
      return;
    }

    MsgLog(__name__(), info, "print_file()\nContent of the file: " << m_fname);

    // open file
    std::ifstream in(m_fname.c_str());
    if (not in.good()) { MsgLog(__name__(), error, "Failed to open file: "+m_fname); return; }
  
    // read and dump all fields
    //std::string s; while(in) { in >> s; cout << s << " "; }

    // read and dump all strings
    std::string str; 
    while(getline(in,str)) cout << str << '\n';
    cout << '\n';
    //close file
    in.close();
}

//-----------------------------

template <typename TDATA>
void ArrayIO<TDATA, NDIM>::print_ndarray()
{
    //const ndarray<const TDATA, NDIM> nda = get_ndarray();
    if (! p_nda) load_ndarray();
    if (! p_nda) return;

    std::stringstream smsg; 
    smsg << "Print ndarray<" << strDataType<TDATA>() 
         << "," << ndim()
         << "> of size=" << p_nda->size()
         << ":\n" << *p_nda;
    MsgLog(__name__(), info, smsg.str());
}

//-----------------------------

template <typename TDATA>
std::string ArrayIO<TDATA>::str_ndarray_info()
{
  //const ndarray<const TDATA, NDIM>& nda = get_ndarray();
    if (! p_nda) load_ndarray();
    if (! p_nda) return std::string("ndarray is non-accessible...");

    std::stringstream smsg; 
    smsg << "ndarray<" << std::setw(8) << std::left << strDataType<TDATA>() 
         << "," << ndim()
         << "> of size=" << p_nda->size()
         << ":";
    TDATA* it = p_nda->data();
    for( unsigned i=0; i<min(size_t(10),p_nda->size()); i++ ) smsg << " " << *it++; smsg << " ...";
    //MsgLog(__name__(), info, smsg.str());
    return smsg.str();
}

//-----------------------------

template <typename TDATA>
std::string ArrayIO<TDATA>::str_shape()
{
  std::stringstream smsg;
  for( unsigned i=0; i<ndim(); i++ ) {
    smsg << m_shape[i]; if (i<ndim()-1) smsg << ", ";
  }
  return smsg.str();
}

//-----------------------------

template <typename TDATA>
void ArrayIO<TDATA>::save_ndarray(const ndarray<const TDATA, NDIM>& nda, 
                                    const std::string& fname,
                                    const std::vector<std::string>& vcoms, 
	                            const unsigned& print_bits)
{
    const unsigned ndim = NDIM;
    std::string str_dtype = strDataType<TDATA>();
    std::stringstream sstype; sstype << "ndarray<" << str_dtype 
                                     << "," << ndim << ">";

    if (print_bits & 1) {
        std::stringstream smsg; 
        smsg << "Save " << sstype.str()
             << " of size=" << nda.size()
             << " in file: " << fname;
        MsgLog(__name__(), info, smsg.str());
    }

    // open file
    std::ofstream out(fname.c_str());
    if (not out.good()) { 
       if(print_bits) MsgLog(__name__(), error, "Failed to open output file: " + fname); 
       return; 
    }
  
    // write comments if available
    if (!vcoms.empty()) {
      for(vector<string>::const_iterator it = vcoms.begin(); it != vcoms.end(); it++)
        out << "# " << *it << '\n';
      out << '\n';
    }

    // write permanent comments
    out << "# DATE_TIME  " << strTimeStamp() << '\n';
    out << "# AUTHOR     " << strEnvVar("LOGNAME") << '\n';
    out << '\n';

    // write metadata
    out << "# Metadata for " << sstype.str() << '\n';
    out << "# DTYPE    " << str_dtype << '\n';
    out << "# NDIM     " << ndim << '\n';
    //shape_t shape = nda.shape()
    for(unsigned i=0; i<ndim; i++) out << "# DIM:" << i << "    " << nda.shape()[i] << '\n';
    out << '\n';

    // save data
    unsigned nmax_in_line = (ndim>1) ? nda.shape()[ndim-1] : 10; 
    unsigned count_in_line=0; 

    typename ndarray<const TDATA>::iterator it = nda.begin();
    for (; it!=nda.end(); ++it) {
      out << std::setw(10) << *it << " ";
      if( ++count_in_line < nmax_in_line) continue;
          count_in_line = 0;
          out << '\n';
    }

    //close file
    out.close();
}

*/

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
