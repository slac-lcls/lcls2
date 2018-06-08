//---------------------------------------------------
// Adopted to lcls2 on 2018-06-06 by Mikhail Dubrovin
//---------------------------------------------------

#include "../include/ArrayIO.h"
#include "../include/Logger.h" // for MsgLog

//#include "MsgLogger/MsgLogger.h"
//#include <algorithm>
//#include <stdexcept>
#include <fstream>
#include <sstream>   // for stringstream
#include <stdlib.h>  // for atoi
#include <cstring>   // for memcpy
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
  init();
  //_size = 0; // actual size is loaded from metadata
}

//-----------------------------

template <typename TDATA>
ArrayIO<TDATA>::~ArrayIO()
{
  std::stringstream ss;
  MsgLog(__name__(), DEBUG, ss << "DESTRUCTOR for ctor:" << _ctor << ", fname=" << _fname);

  //delete p_nda;
}

//-----------------------------

template <typename TDATA>
void ArrayIO<TDATA>::init()
{
  std::stringstream ss;
  MsgLog(__name__(), DEBUG, ss << "ctor:" << _ctor << ", fname=" << _fname);

  _status = ArrayIO<TDATA>::UNDEFINED;
}

//-----------------------------
/*

template <typename TDATA, unsigned NDIM>
ArrayIO<TDATA, NDIM>::ArrayIO(const std::string& fname
				 ,const shape_t* shape_def
				 ,const TDATA& val_def 
                                 ,const unsigned print_bits) 
  : p_nda(0)
  , m_fname(fname)
  , m_val_def(val_def)
  , m_nda_def(ndarray<const TDATA, NDIM>())
  , m_print_bits(print_bits)
  , m_ctor(1)
{
  init();
  std::memcpy(&m_shape[0], shape_def, c_ndim*sizeof(shape_t));  
  //for(unsigned i=0; i<c_ndim; ++i) m_shape[i] = shape_def[i];
  //cout << "TIOV1: shape_def : [" << shape_def[0] << "," << shape_def[1] << "]\n";
  //cout << "TIOV1: m_shape   : [" << m_shape[0] << "," << m_shape[1] << "]\n";
}

//-----------------------------

template <typename TDATA, unsigned NDIM>
ArrayIO<TDATA, NDIM>::ArrayIO(const std::string& fname
	                         ,const ndarray<const TDATA, NDIM>& nda_def
                                 ,const unsigned print_bits) 
  : p_nda(0)
  , m_fname(fname)
  , m_val_def(0)
  , m_nda_def(nda_def)
  , m_print_bits(print_bits)
  , m_ctor(2)
{
  init();
  m_size = nda_def.size();
  std::memcpy(&m_shape[0], nda_def.shape(), c_ndim*sizeof(shape_t));  
  //std::cout << "XXX m_nda_def in c-tor 2: " << m_nda_def << '\n';
}

//-----------------------------

//-----------------------------

//-----------------------------

template <typename TDATA>
void ArrayIO<TDATA>::load_ndarray()
{
    // if file is not available - create default ndarray
    if ((!file_is_available()) && m_ctor>0) { 
        if( m_print_bits & 4 ) MsgLog(__name__(), warning, "Use default calibration parameters.");
        create_ndarray(true);
        m_status = ArrayIO<TDATA>::DEFAULT; // std::string("used default");
        return; 
    }

    if( m_print_bits & 1 ) MsgLog(__name__(), info, "Load file \"" << m_fname << "\"");

    m_count_str_data = 0;
    m_count_str_comt = 0;
    m_count_data     = 0;

    // open file
    std::ifstream in(m_fname.c_str());
    if (not in.good()) { 
        if(m_print_bits) MsgLog(__name__(), error, "Failed to open file: \"" + m_fname + "\""); 
	m_status = ArrayIO<TDATA, NDIM>::UNREADABLE; // std::string("file is unreadable");
        return;
    }
  
    // read and process all strings
    std::string str; 
    while(getline(in,str)) {
        // cout << str << '\n';

        // 1. parse lines with comments marked by # in the 1st position
        if(str[0] == '#') parse_str_of_comment(str.substr(1));

        // 2. skip empty lines 
        else if (str.find_first_not_of(" ")==string::npos) continue; 

        // 3. parse 1st line and load other data
        else load_data(in,str);
    }

    //close file
    in.close();
    m_status = ArrayIO<TDATA>::LOADED; // std::string("loaded from file");
}

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

template <typename TDATA>
void ArrayIO<TDATA>::parse_str_of_comment(const std::string& str)
{
    m_count_str_comt ++;
    // cout << "comment, str.size()=" << str.size() << '\n';

    std::string field;
    std::stringstream ss(str);

    ss >> field;

    if (field=="DTYPE") { 
      ss >> m_str_type;
      m_enum_type = enumDataTypeForString(m_str_type);

      if( m_print_bits & 32 )
        if (m_enum_type != enumDataType<TDATA>()) {
          std::stringstream smsg; 
          smsg << "(enum) DTYPE in file metadata (" << strDataTypeForEnum(m_enum_type)
               << ") is different from expected (" << strDataTypeForEnum(enumDataType<TDATA>()) << ")";
          MsgLog(__name__(), warning, smsg.str());
        }
    }

    else if (field=="NDIM") {
      ss >> m_ndim;
      if (m_ndim != ndim()) {	
	std::stringstream smsg; 
        smsg << "NDIM in file metadata: " << m_ndim 
             << " is different from declaration: " << ndim();
        MsgLog(__name__(), warning, smsg.str());
        //throw std::runtime_error(smsg.str());
      }
    }

    else if (field.substr(0,4)=="DIM:") { 
        //cout << "field.substr(0,4)" << field.substr(0,4) << "field[4]" << field[4] << endl;
        int dim = atoi(&field[4]); 
	shape_t val;    
        ss >> val;

	if (m_ctor == 0) { // get ndarray shape and size from metadata
            m_shape[dim] = val;
	    m_size *= val;
	    // cout << "Current m_size = " << m_size << '\n';
	}
	else if (m_shape[dim] !=val) { // check that metadata is consistent with expected
	   std::stringstream smsg; 
	   smsg << "NDArray metadata shape field " << field
                << " = " << val 
	        << " is different from expected " << m_shape[dim] 
                << " in file " << m_fname
	        << "\nCheck that calibration file has expected shape and data...";
           MsgLog(__name__(), warning, smsg.str());
           //throw std::runtime_error(smsg.str());
	}
    }

    else 
      //MsgLog(__name__(), info, "Ignore comment: " << str );
      return;
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
void ArrayIO<TDATA, NDIM>::load_data(std::ifstream& in, const std::string& str)
{
    if (! m_count_str_data++) create_ndarray();

    // parse the 1st string
    TDATA val;
    TDATA* it=p_data; 

    std::stringstream ss(str);
    while (ss >> val and m_count_data != m_size) { 
      *it++ = val;
      ++m_count_data;
      //if( ndim()==1 ) cout << "count:data = " << m_count_data << " : " << val << '\n';
    }

    // load all data by the end
    while(in >> val and m_count_data != m_size) {
      *it++ = val;
      ++m_count_data;
      //if( ndim()==1 ) cout << "count:data = " << m_count_data << " : " << val << '\n';
    }

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
