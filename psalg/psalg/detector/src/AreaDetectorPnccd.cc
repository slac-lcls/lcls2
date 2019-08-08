
//#include <string>
//#include <iostream> // for cout, puts
#include <iomanip>    // for setw, hex, setprecision, right
#include <sstream>    // for stringstream

#include "psalg/detector/AreaDetectorPnccd.hh"
#include "psalg/utils/Logger.hh" // for MSG

using namespace std;
using namespace psalg;

namespace detector {

//-----------------------------

AreaDetectorPnccd::AreaDetectorPnccd(const std::string& detname, XtcData::ConfigIter& configo)
  : AreaDetector(detname, configo) {
  MSG(DEBUG, "In c-tor AreaDetectorPnccd(detname, configo) for " << detname);
  //process_config();
}

AreaDetectorPnccd::AreaDetectorPnccd(const std::string& detname)
  : AreaDetector(detname) {
  MSG(DEBUG, "In c-tor AreaDetectorPnccd(detname) for " << detname);
}

AreaDetectorPnccd::AreaDetectorPnccd()
  : AreaDetector() {
  MSG(DEBUG, "In default c-tor AreaDetectorPnccd()");
}

AreaDetectorPnccd::~AreaDetectorPnccd() {
  MSG(DEBUG, "In d-tor AreaDetectorPnccd for " << detname());
}

void AreaDetectorPnccd::_class_msg(const std::string& msg) {
  MSG(INFO, "In AreaDetectorPnccd::"<< msg);
}

  /*
const shape_t* AreaDetectorPnccd::shape(const event_t&) {
  _class_msg("shape(...)");
  return &AreaDetector::_shape[0];
  //return &_shape[0];
}

const size_t AreaDetectorPnccd::ndim(const event_t&) {
  _class_msg("ndim(...)");
  return 0;
}

const size_t AreaDetectorPnccd::size(const event_t&) {
  _class_msg("size(...)");
  return 123;
}

const shape_t* AreaDetectorPnccd::shape() {
  _class_msg("shape(...)");
  return &AreaDetector::_shape[0];
}

const size_t AreaDetectorPnccd::size() {
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

void AreaDetectorPnccd::set_indexes_config(XtcData::ConfigIter& configiter) {
  _pconfig = &configiter;
  XtcData::Names& names = configNames(configiter);
  MSG(DEBUG, str_config_names(configiter));

  for (unsigned m=0; m<MAX_NUMBER_OF_MODULES; m++) {
    sprintf(&_cbuf1[m][0], "frame%d_frameNumber", m);
    sprintf(&_cbuf2[m][0], "frame%d_timeStampHi", m);
    sprintf(&_cbuf3[m][0], "frame%d_timeStampLo", m);
    sprintf(&_cbuf4[m][0], "frame%d_specialWord", m);
    sprintf(&_cbuf5[m][0], "frame%d_data", m);
    sprintf(&_cbuf6[m][0], "frame%d__data", m);
  }

  for (unsigned i=0; i<names.num(); i++) {
      const char* cname = names.get(i).name();

      //XtcData::Name::DataType itype = name.type();
      //printf("\n  %02d name: %-32s rank: %d type: %d el.size %02d",
      //       i, cname, name.rank(), itype, Name::get_element_size(itype));

      if     (strcmp(cname, "Version")              ==0) _Version  = i;
      else if(strcmp(cname, "TypeId")               ==0) _TypeId   = i;
      else if(strcmp(cname, "numLinks")             ==0) _numLinks = i;
      else if(strcmp(cname, "numChannels")          ==0) _numChannels = i;
      else if(strcmp(cname, "camexMagic")           ==0) _camexMagic = i;
      else if(strcmp(cname, "numRows")              ==0) _numRows = i;
      else if(strcmp(cname, "numSubmoduleRows")     ==0) _numSubmoduleRows = i;
      else if(strcmp(cname, "payloadSizePerLink")   ==0) _payloadSizePerLink = i;
      else if(strcmp(cname, "numSubmoduleChannels") ==0) _numSubmoduleChannels = i;
      else if(strcmp(cname, "numSubmodules")        ==0) _numSubmodules = i;
      else if(strcmp(cname, "info_shape")           ==0) _info_shape = i;
      else if(strcmp(cname, "timingFName_shape")    ==0) _timingFName_shape = i;
  }

  // derived values
  maxModulesPerDetector = MAX_NUMBER_OF_MODULES;
  numberOfModules       = numSubmodules();
  numberOfRows          = numSubmoduleRows();
  numberOfColumns       = numSubmoduleChannels();
  numberOfPixels        = numberOfModules * numberOfRows * numberOfColumns;
}

//-------------------

void AreaDetectorPnccd::set_indexes_data(XtcData::DataIter& dataiter) {
    ConfigIter& configo = *_pconfig;
    NamesLookup& namesLookup = configo.namesLookup();
    DescData& descdata = dataiter.desc_value(namesLookup);
    Names& names = descdata.nameindex().names();

    printf("\n-------- %d Names and values for data --------\n", names.num());

    for(unsigned i=0; i<names.num(); i++) {
        Name& name = names.get(i);
        printf("  %02d name: %-32s rank: %d type: %d : %s\n", i, name.name(), name.rank(), name.type(), name.str_type());

        const char* cname = name.name();

        if     (strcmp(cname, "Version")    ==0) _data_Version  = i;
        else if(strcmp(cname, "TypeId")     ==0) _data_TypeId   = i;
        else if(strcmp(cname, "numLinks")   ==0) _numLinks      = i;
        else if(strcmp(cname, "frame_shape")==0) _frame_shape   = i;

        for(unsigned m=0; m<MAX_NUMBER_OF_MODULES; m++) {
            if(strcmp(cname, _cbuf1[m])==0) {_frameNumber[m] = i; break;}
            if(strcmp(cname, _cbuf2[m])==0) {_timeStampHi[m] = i; break;}
            if(strcmp(cname, _cbuf3[m])==0) {_timeStampLo[m] = i; break;}
            if(strcmp(cname, _cbuf4[m])==0) {_specialWord[m] = i; break;}
            if(strcmp(cname, _cbuf5[m])==0) {_data[m]        = i; break;}
            if(strcmp(cname, _cbuf6[m])==0) {__data[m]       = i; break;}
        }
    }
}

//-------------------

const void AreaDetectorPnccd::print_config() {

  std::cout << "\n\n==== Attributes of configuration ====\n";
  std::cout << "detname              : " << detname()              << '\n';
  std::cout << "dettype              : " << dettype()              << '\n';
  std::cout << "Version              : " << Version()              << '\n';
  std::cout << "TypeId               : " << TypeId()               << '\n';
  std::cout << "camexMagic           : " << camexMagic()           << '\n';
  std::cout << "numChannels          : " << numChannels()          << '\n';
  std::cout << "numRows              : " << numRows()              << '\n';
  std::cout << "numLinks             : " << numLinks()             << '\n';
  std::cout << "numSubmodules        : " << numSubmodules()        << '\n';
  std::cout << "numSubmoduleRows     : " << numSubmoduleRows()     << '\n';
  std::cout << "numSubmoduleChannels : " << numSubmoduleChannels() << '\n';
  std::cout << "payloadSizePerLink   : " << payloadSizePerLink()   << '\n';
  std::cout << "info_shape        nda: " << (NDArray<int64_cfg_t>)info_shape() << '\n';
  std::cout << "timingFName_shape nda: " << (NDArray<int64_cfg_t>)timingFName_shape() << '\n';

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

const void AreaDetectorPnccd::print_data() {
    std::cout << "\n==== Data ====\n";
    _class_msg("print_data() is DEPRECATED");
}

//-------------------

const void AreaDetectorPnccd::print_data(XtcData::DataIter& di) {
    std::cout << "\n==== Data ====\n";
    std::cout << "Version      : " << data_Version(di) << '\n';
    std::cout << "TypeId       : " << data_TypeId(di)  << '\n';
    std::cout << "numLinks     : " << data_numLinks(di) << '\n';
    std::cout << "frame_shape  : " << (NDArray<int64_cfg_t>)frame_shape(di)  << '\n';

    for(unsigned m=0; m<MAX_NUMBER_OF_MODULES; m++) std::cout << _cbuf1[m] << ": " << frameNumber(di ,m) << '\n';
    for(unsigned m=0; m<MAX_NUMBER_OF_MODULES; m++) std::cout << _cbuf2[m] << ": " << timeStampHi(di ,m) << '\n';
    for(unsigned m=0; m<MAX_NUMBER_OF_MODULES; m++) std::cout << _cbuf3[m] << ": " << timeStampLo(di ,m) << '\n';
    for(unsigned m=0; m<MAX_NUMBER_OF_MODULES; m++) std::cout << _cbuf4[m] << ": " << specialWord(di ,m) << '\n';
    for(unsigned m=0; m<MAX_NUMBER_OF_MODULES; m++) std::cout << _cbuf5[m] << ": " << (NDArray<uint16_t>)data2d(di ,m) << '\n';
    for(unsigned m=0; m<MAX_NUMBER_OF_MODULES; m++) std::cout << _cbuf6[m] << ": " << (NDArray<uint16_t>)data1d(di ,m) << '\n';
}

//-------------------

void AreaDetectorPnccd::_panel_id(std::ostream& os, const int ind) {
  os << setfill('0') << std::setw(4) << ind;
    //os << std::hex << moduleVersion[ind] << '-' 
    //   << std::hex << _firmwareVersion[ind] << '-' 
    //   << std::hex << setfill('0')  << serialNumber[ind];
    //// << std::hex << std::setw(10) << std::setprecision(8) << std::right<< setfill('0')
}

//-------------------

void AreaDetectorPnccd::detid(std::ostream& os, const int ind) {
  //os << "panel index="  << std::setw(2) << ind << " numberOfModules=" << numberOfModules << " == ";
  assert(ind<MAX_NUMBER_OF_MODULES);
  assert(numberOfModules>0);
  assert(numberOfModules<=MAX_NUMBER_OF_MODULES);
  if(ind > -1) return _panel_id(os, ind);
  for(int i=0; i<numberOfModules; i++) {
    if(i) os << '_';
    _panel_id(os, i);
  }
}

//-------------------

NDArray<raw_pnccd_t>& AreaDetectorPnccd::raw(XtcData::DescData& ddata) {
    if(_ind_data < 0) _set_index_data(ddata, "frame");
    raw_pnccd_t* pdata = ddata.get_array<raw_pnccd_t>(_ind_data).data();
    _raw.set_shape(shape(), ndim());
    _raw.set_data_buffer(pdata);
    return _raw;
}

//-------------------

NDArray<raw_pnccd_t>& AreaDetectorPnccd::raw(XtcData::DataIter& datao) {
    DescData&  ddata = datao.desc_value(_pconfig->namesLookup());
    return raw(ddata);
}

//-------------------

/* implemented in AreaDetector
std::string AreaDetectorPnccd::detid(const int& ind) {
  std::stringstream ss;
  detid(ss, ind);
  return ss.str();
}
*/

//-------------------
//-------------------

} // namespace detector

//-----------------------------
