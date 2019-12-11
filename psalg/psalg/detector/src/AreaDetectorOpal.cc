
//#include <string>
//#include <iostream> // for cout, puts
#include <iomanip>    // for setw, hex, setprecision, right
#include <sstream>    // for stringstream
#include <cstring>    // memcpy

#include "psalg/detector/AreaDetectorOpal.hh"
#include "psalg/utils/Logger.hh" // for MSG

using namespace std;
using namespace psalg;

namespace detector {

//-----------------------------

AreaDetectorOpal::AreaDetectorOpal(const std::string& detname, XtcData::ConfigIter& ci)
  : AreaDetector(detname, ci) {
  MSG(DEBUG, "In c-tor AreaDetectorOpal(detname, configo) for " << detname);
  //process_config();
  _set_indexes_config(ci);
}

AreaDetectorOpal::AreaDetectorOpal(const std::string& detname)
  : AreaDetector(detname) {
  MSG(DEBUG, "In c-tor AreaDetectorOpal(detname) for " << detname);
}

AreaDetectorOpal::AreaDetectorOpal()
  : AreaDetector() {
  MSG(DEBUG, "In default c-tor AreaDetectorOpal()");
}

AreaDetectorOpal::~AreaDetectorOpal() {
  MSG(DEBUG, "In d-tor AreaDetectorOpal for " << detname());
}

void AreaDetectorOpal::_class_msg(const std::string& msg) {
  MSG(INFO, "In AreaDetectorOpal::"<< msg);
}

  /*
const shape_t* AreaDetectorOpal::shape(const event_t&) {
  _class_msg("shape(...)");
  return &AreaDetector::_shape[0];
  //return &_shape[0];
}

const size_t AreaDetectorOpal::ndim(const event_t&) {
  _class_msg("ndim(...)");
  return 0;
}

const size_t AreaDetectorOpal::size(const event_t&) {
  _class_msg("size(...)");
  return 123;
}

const shape_t* AreaDetectorOpal::shape() {
  _class_msg("shape(...)");
  return &AreaDetector::_shape[0];
}

const size_t AreaDetectorOpal::size() {
  _class_msg("size(...)");
  return 123;
}
  */


/// access to calibration constants
/*

const NDArray<common_mode_t>&   common_mode      (const event_t&) = 0;
const NDArray<pedestals_t>&     pedestals        (const event_t&) = 0;
const NDArray<pixel_rms_t>&     rms              (const event_t&) = 0;
const NDArray<pixel_status_t>&  status           (const event_t&) = 0;
const NDArray<pixel_gain_t>&    gain             (const event_t&) = 0;
const NDArray<pixel_offset_t>&  offset           (const event_t&) = 0;
const NDArray<pixel_bkgd_t>&    background       (const event_t&) = 0;
const NDArray<pixel_mask_t>&    mask_calib       (const event_t&) = 0;
const NDArray<pixel_mask_t>&    mask_from_status (const event_t&) = 0;
const NDArray<pixel_mask_t>&    mask_edges       (const event_t&, const size_t& nnbrs=8) = 0;
const NDArray<pixel_mask_t>&    mask_neighbors   (const event_t&, const size_t& nrows=1, const size_t& ncols=1) = 0;
const NDArray<pixel_mask_t>&    mask             (const event_t&, const size_t& mbits=0177777) = 0;
const NDArray<pixel_mask_t>&    mask             (const event_t&, const bool& calib=true,
					                          const bool& sataus=true,
                                                                  const bool& edges=true,
						                  const bool& neighbors=true) = 0;

/// access to raw, calibrated data, and image
const NDArray<raw_t>&   raw  (const event_t&) = 0;
const NDArray<calib_t>& calib(const event_t&) = 0;
const NDArray<image_t>& image(const event_t&) = 0;
const NDArray<image_t>& image(const event_t&, const NDArray<image_t>& nda) = 0;
const NDArray<image_t>& array_from_image(const event_t&, const NDArray<image_t>&) = 0;
void move_geo(const event_t&, const pixel_size_t& dx,  const pixel_size_t& dy,  const pixel_size_t& dz) = 0;
void tilt_geo(const event_t&, const tilt_angle_t& dtx, const tilt_angle_t& dty, const tilt_angle_t& dtz) = 0;

/// access to geometry
const geometry_t* geometry(const event_t&) = 0;
const NDArray<pixel_idx_t>&   indexes    (const event_t&, const size_t& axis=0) = 0;
const NDArray<pixel_coord_t>& coords     (const event_t&, const size_t& axis=0) = 0;
const NDArray<pixel_size_t>&  pixel_size (const event_t&, const size_t& axis=0) = 0;
const NDArray<pixel_size_t>&  image_xaxis(const event_t&) = 0;
const NDArray<pixel_size_t>&  image_yaxis(const event_t&) = 0;
*/

//-------------------

void AreaDetectorOpal::_set_indexes_config(XtcData::ConfigIter& configiter) {

  if(!_pconfit) return;
  _pconfit = &configiter;

  XtcData::Names& names = configNames(configiter);
  MSG(DEBUG, str_config_names(configiter));

  for (unsigned i=0; i<names.num(); i++) {
      const char* cname = names.get(i).name();

      //XtcData::Name::DataType itype = name.type();
      //printf("\n  %02d name: %-32s rank: %d type: %d el.size %02d",
      //       i, cname, name.rank(), itype, Name::get_element_size(itype));

      if     (strcmp(cname, "Version")                         ==0) _Version                         = i;
      else if(strcmp(cname, "TypeId")                          ==0) _TypeId                          = i;
      else if(strcmp(cname, "defect_pixel_correction_enabled") ==0) _defect_pixel_correction_enabled = i;
      else if(strcmp(cname, "number_of_defect_pixels")         ==0) _number_of_defect_pixels         = i;
      else if(strcmp(cname, "output_offset")                   ==0) _output_offset                   = i;
      else if(strcmp(cname, "gain_percent")                    ==0) _gain_percent                    = i;
      else if(strcmp(cname, "Column_Pixels")                   ==0) _Column_Pixels                   = i;
      else if(strcmp(cname, "Row_Pixels")                      ==0) _Row_Pixels                      = i;
      else if(strcmp(cname, "Mirroring")                       ==0) _Mirroring                       = i;
      else if(strcmp(cname, "output_mirroring")                ==0) _output_mirroring                = i;
      else if(strcmp(cname, "vertical_binning")                ==0) _vertical_binning                = i;
      else if(strcmp(cname, "Depth")                           ==0) _Depth                           = i;
      else if(strcmp(cname, "Output_LUT_Size")                 ==0) _Output_LUT_Size                 = i;
      else if(strcmp(cname, "Binning")                         ==0) _Binning                         = i;
      else if(strcmp(cname, "output_resolution")               ==0) _output_resolution               = i;
      else if(strcmp(cname, "output_resolution_bits")          ==0) _output_resolution_bits          = i;
      else if(strcmp(cname, "vertical_remapping")              ==0) _vertical_remapping              = i;
      else if(strcmp(cname, "LUT_Size")                        ==0) _LUT_Size                        = i;
      else if(strcmp(cname, "output_lookup_table_enabled")     ==0) _output_lookup_table_enabled     = i;
      else if(strcmp(cname, "black_level")                     ==0) _black_level                     = i;
      else if(strcmp(cname, "output_lookup_table")             ==0) _output_lookup_table             = i;
  }

  // derived values
  maxModulesPerDetector = 1;
  numberOfModules       = 1;
  numberOfRows          = Row_Pixels();
  numberOfColumns       = Column_Pixels();
  numberOfPixels        = numberOfModules * numberOfRows * numberOfColumns;
}

//-------------------

void AreaDetectorOpal::_set_indexes_data(XtcData::DataIter& dataiter) {
    if(_pdatait) return;
    _pdatait = &dataiter;

    ConfigIter& configo = *_pconfit;
    NamesLookup& namesLookup = configo.namesLookup();
    DescData& descdata = dataiter.desc_value(namesLookup);
    Names& names = descdata.nameindex().names();

    printf("\n-------- %d Names and values for data --------\n", names.num());

    for(unsigned i=0; i<names.num(); i++) {
        Name& name = names.get(i);
        printf("  %02d name: %-32s rank: %d type: %d : %s\n", i, name.name(), name.rank(), name.type(), name.str_type());

        const char* cname = name.name();
        if     (strcmp(cname, "data_Version")    ==0) _data_Version    = i;
        else if(strcmp(cname, "data_TypeId")     ==0) _data_TypeId     = i;
        else if(strcmp(cname, "height")          ==0) _height          = i;
        else if(strcmp(cname, "width")           ==0) _width           = i;
        else if(strcmp(cname, "depth")           ==0) _depth           = i;
        else if(strcmp(cname, "offset")          ==0) _offset          = i;
        else if(strcmp(cname, "depth_bytes")     ==0) _depth_bytes     = i;
        else if(strcmp(cname, "_int_pixel_data") ==0) __int_pixel_data = i;
        else if(strcmp(cname, "data8")           ==0) _data8           = i;
        else if(strcmp(cname, "data16")          ==0) _data16          = i;
    }
}

//-------------------

const void AreaDetectorOpal::print_config() {

  std::cout << "\n\n==== Attributes of configuration ====\n";
  std::cout << "detname                         : " << detname()                               << '\n';
  std::cout << "dettype                         : " << dettype()                               << '\n';
  std::cout << "Version                         : " << Version()                               << '\n';
  std::cout << "TypeId                          : " << TypeId()                                << '\n';
  std::cout << "defect_pixel_correction_enabled : " << defect_pixel_correction_enabled()       << '\n';
  std::cout << "number_of_defect_pixels         : " << number_of_defect_pixels()               << '\n';
  std::cout << "output_offset                   : " << output_offset()                         << '\n';
  std::cout << "gain_percent                    : " << gain_percent()                          << '\n';
  std::cout << "Column_Pixels                   : " << Column_Pixels()                         << '\n';
  std::cout << "Row_Pixels                      : " << Row_Pixels()                            << '\n';
  std::cout << "Mirroring                       : " << Mirroring()                             << '\n';
  std::cout << "output_mirroring                : " << output_mirroring()                      << '\n';
  std::cout << "vertical_binning                : " << vertical_binning()                      << '\n';
  std::cout << "Depth                           : " << Depth()                                 << '\n';
  std::cout << "Output_LUT_Size                 : " << Output_LUT_Size()                       << '\n';
  std::cout << "Binning                         : " << Binning()                               << '\n';
  std::cout << "output_resolution               : " << output_resolution()                     << '\n';
  std::cout << "output_resolution_bits          : " << output_resolution_bits()                << '\n';
  std::cout << "vertical_remapping              : " << vertical_remapping()                    << '\n';
  std::cout << "LUT_Size                        : " << LUT_Size()                              << '\n';
  std::cout << "output_lookup_table_enabled     : " << output_lookup_table_enabled()           << '\n';
  std::cout << "black_level                     : " << black_level()                           << '\n';
  std::cout << "output_lookup_table             : " << (NDArray<uint16_t>)output_lookup_table()<< '\n';

  std::cout << "\n==== Derived values ====\n";
  std::cout << "numberOfModules      : " << numberOfModules        << '\n';
  std::cout << "numberOfRows         : " << numberOfRows           << '\n';
  std::cout << "numberOfColumns      : " << numberOfColumns        << '\n';
  std::cout << "numPixels            : " << numberOfPixels         << '\n';
  std::cout << "detid                : " << AreaDetector::detid()  << '\n';
  std::cout << "ndim()               : " << ndim()                 << '\n';
  std::cout << "size()               : " << size()                 << '\n';
}

//-------------------

const void AreaDetectorOpal::print_config_indexes() {
  std::cout << "\n\n==== indecses for names in configuration ====\n"; 
  std::cout << setfill(' ');
  std::cout << "Version                        : " << std::right << std::setw(2) << _Version                         << '\n';
  std::cout << "TypeId                         : " << std::right << std::setw(2) << _TypeId                          << '\n';
  std::cout << "defect_pixel_correction_enabled: " << std::right << std::setw(2) << _defect_pixel_correction_enabled << '\n';
  std::cout << "number_of_defect_pixels        : " << std::right << std::setw(2) << _number_of_defect_pixels         << '\n';
  std::cout << "output_offset                  : " << std::right << std::setw(2) << _output_offset                   << '\n';
  std::cout << "gain_percent                   : " << std::right << std::setw(2) << _gain_percent                    << '\n';
  std::cout << "Column_Pixels                  : " << std::right << std::setw(2) << _Column_Pixels                   << '\n';
  std::cout << "Row_Pixels                     : " << std::right << std::setw(2) << _Row_Pixels                      << '\n';
  std::cout << "Mirroring                      : " << std::right << std::setw(2) << _Mirroring                       << '\n';
  std::cout << "output_mirroring               : " << std::right << std::setw(2) << _output_mirroring                << '\n';
  std::cout << "vertical_binning               : " << std::right << std::setw(2) << _vertical_binning                << '\n';
  std::cout << "Depth                          : " << std::right << std::setw(2) << _Depth                           << '\n';
  std::cout << "Output_LUT_Size                : " << std::right << std::setw(2) << _Output_LUT_Size                 << '\n';
  std::cout << "Binning                        : " << std::right << std::setw(2) << _Binning                         << '\n';
  std::cout << "output_resolution              : " << std::right << std::setw(2) << _output_resolution               << '\n';
  std::cout << "output_resolution_bits         : " << std::right << std::setw(2) << _output_resolution_bits          << '\n';
  std::cout << "vertical_remapping             : " << std::right << std::setw(2) << _vertical_remapping              << '\n';
  std::cout << "LUT_Size                       : " << std::right << std::setw(2) << _LUT_Size                        << '\n';
  std::cout << "output_lookup_table_enabled    : " << std::right << std::setw(2) << _output_lookup_table_enabled     << '\n';
  std::cout << "black_level                    : " << std::right << std::setw(2) << _black_level                     << '\n';
  std::cout << "output_lookup_table            : " << std::right << std::setw(2) << _output_lookup_table             << '\n';
}

//-------------------
/**
#define LONG_STRING1(T)\
   for(unsigned m=0; m<MAX_NUMBER_OF_MODULES; m++)\
     std::cout << std::left << std::setw(20) << T << '\n';
*/

const void AreaDetectorOpal::print_data(XtcData::DataIter& di) {

  if(!_pdatait) _set_indexes_data(di);

  std::cout << "\n==== Data ====\n";
  std::cout << "data_Version      : " << data_Version(di) << '\n';
  std::cout << "data_TypeId       : " << data_TypeId(di)  << '\n';
  std::cout << "height            : " << height(di)       << '\n';
  std::cout << "width             : " << width(di)        << '\n';
  std::cout << "depth             : " << depth(di)        << '\n';
  std::cout << "offset            : " << offset(di)       << '\n';
  std::cout << "depth_bytes       : " << depth_bytes(di)  << '\n';

  std::cout << "_int_pixel_data   : " << (NDArray<uint8_t>)_int_pixel_data(di) << '\n';
  std::cout << "data8             : " << (NDArray<uint8_t>)data8(di)           << '\n';
  std::cout << "data16            : " << (NDArray<uint16_t>)data16(di)         << '\n';
}

//-------------------
/*
#define LONG_STRING2(T,I)\
 for(unsigned m=0; m<MAX_NUMBER_OF_MODULES; m++)\
   std::cout << std::left << std::setw(20) << T << ": " << std::right << std::setw(2) << I << '\n';

 LONG_STRING2(_cbuf1[m], _frameNumber[m])
*/

const void AreaDetectorOpal::print_data_indexes(XtcData::DataIter& di) {

  if(!_pdatait) _set_indexes_data(di);

  std::cout << "\n\n==== indecses for names in data ====\n";
  std::cout << setfill(' ');
  std::cout << "Version         : " << std::right << std::setw(2) << _Version         << '\n';
  std::cout << "TypeId          : " << std::right << std::setw(2) << _TypeId          << '\n';
  std::cout << "height          : " << std::right << std::setw(2) << _height          << '\n';
  std::cout << "width           : " << std::right << std::setw(2) << _width           << '\n';
  std::cout << "depth           : " << std::right << std::setw(2) << _depth           << '\n';
  std::cout << "offset          : " << std::right << std::setw(2) << _offset          << '\n';
  std::cout << "depth_bytes     : " << std::right << std::setw(2) << _depth_bytes     << '\n';
  std::cout << "_int_pixel_data : " << std::right << std::setw(2) << __int_pixel_data << '\n';
  std::cout << "data8           : " << std::right << std::setw(2) << _data8           << '\n';
  std::cout << "data16          : " << std::right << std::setw(2) << _data16          << '\n';
}

//-------------------

void AreaDetectorOpal::_panel_id(std::ostream& os, const int ind) {
  os << setfill('0') << std::setw(4) << ind;
    //os << std::hex << moduleVersion[ind] << '-' 
    //   << std::hex << _firmwareVersion[ind] << '-' 
    //   << std::hex << setfill('0')  << serialNumber[ind];
    //// << std::hex << std::setw(10) << std::setprecision(8) << std::right<< setfill('0')
}

//-------------------

void AreaDetectorOpal::detid(std::ostream& os, const int ind) {
     //os << "panel index="  << std::setw(2) << ind << " numberOfModules=" << numberOfModules << " == ";
//--  assert(ind<MAX_NUMBER_OF_MODULES);
//--  assert(numberOfModules>0);
//--  assert(numberOfModules<=MAX_NUMBER_OF_MODULES);
//--  if(ind > -1) return _panel_id(os, ind);
//--  for(int i=0; i<numberOfModules; i++) {
//--    if(i) os << '_';
//--    _panel_id(os, i);
//--  }
      _panel_id(os, ind);
}

//-------------------

void AreaDetectorOpal::_make_raw(XtcData::DescData& dd) {
  _raw.reserve_data_buffer(size());
  _raw.set_shape(shape(), ndim());
  // copy segment data of four Array to single NDArray 

  //  for(unsigned m=0; m<MAX_NUMBER_OF_MODULES; m++) {
  //    Array<opal_raw_t> a = dd.get_array<opal_raw_t>(__data[m]);
  //    opal_raw_t* pdata = a.data();
  //    memcpy(&_raw(m,0,0), pdata, sizeof(opal_raw_t) * a.num_elem());
  //  }
}

//-------------------

NDArray<opal_raw_t>& AreaDetectorOpal::raw(XtcData::DescData& dd) {
  _make_raw(dd);
  return _raw;
}

//-------------------

NDArray<opal_calib_t>& AreaDetectorOpal::calib(XtcData::DescData& dd) {
  _make_raw(dd);
  unsigned n = size();
  _calib.reserve_data_buffer(n);
  _calib.set_shape(shape(), ndim());

  opal_raw_t*   pr = _raw.data();
  opal_raw_t*   pr_last = &pr[n];
  opal_calib_t* pc = _calib.data();
  const opal_pedestals_t* pp = (*_peds).const_data();
  for(; pr<pr_last; pr++, pc++, pp++) *pc = (opal_calib_t)*pr - (opal_calib_t)*pp;
  return _calib;
}

//-------------------

void AreaDetectorOpal::load_calib_constants() {
  set_calibtype("pedestals");
  const NDArray<opal_pedestals_t>& a = pedestals_d();
  std::cout << "== det.pedestals_d : " << a << '\n';
  _peds = &a;
}

//-------------------

/* implemented in AreaDetector
std::string AreaDetectorOpal::detid(const int& ind) {
  std::stringstream ss;
  detid(ss, ind);
  return ss.str();
}
*/

//-------------------
//-------------------

} // namespace detector

//-----------------------------
