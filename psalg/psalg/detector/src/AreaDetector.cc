
#include <stdio.h>  // for  sprintf, printf( "%lf\n", accum );
#include <iostream> // for cout, puts etc.

#include "psalg/detector/AreaDetector.hh"
#include "psalg/utils/Logger.hh" // for MSG

using namespace std;
using namespace psalg;

//-------------------

namespace detector {

AreaDetector::AreaDetector(const std::string& detname, ConfigIter& config) : 
  Detector(detname, AREA_DETECTOR), _shape(0), _pconfig(&config), _ind_data(-1), _calib_pars(0) {
  MSG(DEBUG, "In c-tor AreaDetector(detname, config) for " << detname);
}

AreaDetector::AreaDetector(const std::string& detname) : 
  Detector(detname, AREA_DETECTOR), _shape(0), _pconfig(NULL), _ind_data(-1), _calib_pars(0) {
  MSG(DEBUG, "In c-tor AreaDetector(detname) for " << detname);
}

AreaDetector::AreaDetector() : 
  Detector(), _shape(0), _pconfig(NULL), _ind_data(-1), _calib_pars(0) {
  MSG(DEBUG, "Default c-tor AreaDetector()");
}

AreaDetector::~AreaDetector() {
  MSG(DEBUG, "In d-tor AreaDetector for " << detname());
  if(_calib_pars) {delete _calib_pars; _calib_pars=0;}
  if(_shape) delete _shape;
}

void AreaDetector::_default_msg(const std::string& msg) const {
  MSG(WARNING, "DEFAULT METHOD AreaDetector::"<< msg << " SHOULD BE RE-IMPLEMENTED IN THE DERIVED CLASS.");
}

//-------------------

void AreaDetector::process_config() {

  ConfigIter& configo = *_pconfig;
  NamesId& namesId = configo.shape().namesId();
  Names& names = configNames(configo);

  MSG(DEBUG, "In AreaDetector::process_config, transition: " << namesId.namesId() << " (0/1 = config/data)\n");
  printf("Names:: detName: %s  detType: %s  detId: %s  segment: %d alg.name: %s\n",
          names.detName(), names.detType(), names.detId(), names.segment(), names.alg().name());

  //DESC_SHAPE(desc_shape, configo, namesLookup);
  DescData& desc_shape = configo.desc_shape();

  //DESC_VALUE(desc_value, configo, namesLookup);
  //DescData& desc_value = configo.desc_value();

  printf("------ ConfigIter %d names and values for detector %s ---------\n", names.num(), names.detName());
  for (unsigned i = 0; i < names.num(); i++) {
      Name& name = names.get(i);
      Name::DataType itype = name.type();
      printf("%02d name: %-32s rank: %d type: %d el.size %02d",
             i, name.name(), name.rank(), itype, Name::get_element_size(itype));

      if (name.rank()==0 and name.type()==Name::INT64) {
  	   printf(" value: %ld\n", desc_shape.get_value<int64_t>(name.name()));
      }
      else if (name.rank()>0) {
          auto array = desc_shape.get_array<uint32_t>(i);
          for (unsigned k=0; k<array.shape()[0]; k++)
              cout << " ==> as uint32_t el:" << k << " value:" << array(k);
	  cout << '\n';
      }
      else printf(" value: TBD\n");
  }
}

//-------------------

void AreaDetector::process_data(XtcData::DataIter& datao) {
    _default_msg("process_data");

    //MSG(DEBUG, "In AreaDetector::process_data");

    ConfigIter& configo = *_pconfig;
    NamesLookup& namesLookup = configo.namesLookup();

    DescData& descdata = datao.desc_value(namesLookup);

    //NameIndex& nameIndex   = descdata.nameindex();
    //ShapesData& shapesData = descdata.shapesdata();
    //NamesId& namesId       = shapesData.namesId();
    Names& names           = descdata.nameindex().names();

    //MSG(DEBUG, str_config_names(configo).c_str());

    printf("------ %d Names and values for data ---------\n", names.num());
    for (unsigned i = 0; i < names.num(); i++) {
        Name& name = names.get(i);
        printf("%02d name: %-32s rank: %d type: %d", i, name.name(), name.rank(), name.type());
        if (name.rank()==0 and name.type()==Name::INT64) {
	  printf(" value %ld\n", descdata.get_value<int64_t>(name.name()));
        }
        else if (name.rank()>0) {
	  uint16_t *data = descdata.get_array<uint16_t>(i).data();
	  printf(" as uint16 ==> %d %d %d %d %d\n", data[0],data[1],data[2],data[3],data[4]);
        }
	else printf("  ==> TBD\n");
    }
}

//-------------------

const void AreaDetector::print_config() {
    _default_msg("print_config");
}

//-------------------

void AreaDetector::_set_index_data(XtcData::DescData& ddata, const char* dataname) {
    Names& names = ddata.nameindex().names();
    for (unsigned i = 0; i < names.num(); i++) {
      if(strcmp(names.get(i).name(), dataname) == 0) {
        _ind_data = (int)i; 
        //MSG(DEBUG, "  ===> dataname: " << dataname << " index: " << _ind_data);
        break;
      }
    }
}

//-------------------

template<typename T>
void AreaDetector::raw(XtcData::DescData& ddata, T*& pdata, const char* dataname) {
    if(_ind_data < 0) _set_index_data(ddata, dataname);
    pdata = ddata.get_array<T>(_ind_data).data();
}

//-------------------

template void AreaDetector::raw<uint16_t>(XtcData::DescData&, uint16_t*&, const char*); 
//template void AreaDetector::raw<int16_t> (XtcData::DescData&, int16_t*&, const char*); 
//template void AreaDetector::raw<int8_t>  (XtcData::DescData&, int8_t*&, const char*); 

//-------------------

template<typename T>
void AreaDetector::raw(XtcData::DataIter& datao, T*& pdata, const char* dataname) {
  //ConfigIter& configo = *_pconfig;
  //NamesLookup& namesLookup = configo.namesLookup();
    DescData& ddata = datao.desc_value(_pconfig->namesLookup());
    raw<T>(ddata, pdata, dataname);
}

//-------------------

  template void AreaDetector::raw<uint16_t>(XtcData::DataIter&, uint16_t*&, const char*); 
//template void AreaDetector::raw<int16_t> (XtcData::DataIter&, int16_t*&, const char*); 
//template void AreaDetector::raw<int8_t>  (XtcData::DataIter&, int8_t*&, const char*); 

//-------------------

template<typename T>
void AreaDetector::raw(XtcData::DescData& ddata, NDArray<T>& nda, const char* dataname) {
  //if(_ind_data < 0) _set_index_data(ddata, dataname);
  //T* pdata = ddata.get_array<T>(_ind_data).data();
    T* pdata = 0;
    raw<T>(ddata, pdata, dataname);
    nda.set_shape(shape(), ndim());
    nda.set_data_buffer(pdata);
}

//-------------------

  template void AreaDetector::raw<uint16_t>(XtcData::DescData&, NDArray<uint16_t>&, const char*); 
//template void AreaDetector::raw<int16_t> (XtcData::DescData&, NDArray<int16_t>&, const char*); 
//template void AreaDetector::raw<int8_t>  (XtcData::DescData&, NDArray<int8_t>&, const char*); 

//-------------------

template<typename T>
void AreaDetector::raw(XtcData::DataIter& datao, NDArray<T>& nda, const char* dataname) {
    DescData& ddata = datao.desc_value(_pconfig->namesLookup());
    raw<T>(ddata, nda, dataname);
}

//-------------------

  template void AreaDetector::raw<uint16_t>(XtcData::DataIter&, NDArray<uint16_t>&, const char*); 
//template void AreaDetector::raw<int16_t> (XtcData::DataIter&, NDArray<int16_t>&, const char*); 
//template void AreaDetector::raw<int8_t>  (XtcData::DataIter&, NDArray<int8_t>&, const char*); 

//-------------------
//-------------------
//-------------------
//-------------------

void AreaDetector::detid(std::ostream& os, const int ind) {
  //_default_msg("detid(std::ostream& os,...)");
  os << "default_area_detector_id";
}

//-------------------

std::string AreaDetector::detid(const int ind) {
  //_default_msg("detid(...) returns string");
  std::stringstream ss;
  detid(ss, ind);
  return ss.str();
}

//-------------------

const size_t AreaDetector::ndim() {
  //_default_msg("ndim(...)");
  return (numberOfModules > 1)? 3 : 2;
}

//-------------------

const size_t AreaDetector::size() {
  //_default_msg("size(...)");
  return (size_t)numberOfPixels;
}

//-------------------

shape_t* AreaDetector::shape() {
  //_default_msg("shape(...)");
  if(!_shape) {
    if (numberOfModules > 1)
          _shape = new shape_t[3]{(shape_t)numberOfModules, (shape_t)numberOfRows, (shape_t)numberOfColumns};
    else  _shape = new shape_t[2]{(shape_t)numberOfRows, (shape_t)numberOfColumns};
    // std::fill_n(_shape, 5, 0); _shape[0]=11;
  }
  return &_shape[0];
  //return _shape;
}

//-------------------

const shape_t* AreaDetector::shape(const event_t& evt) {
  _default_msg("shape(...)");
  if(!_shape) _shape=new shape_t[3]{1,2,3};
  return &_shape[0];
}

const size_t AreaDetector::ndim(const event_t& evt) {
  _default_msg("ndim(...)");
  return 0;
}

const size_t AreaDetector::size(const event_t& evt) {
  _default_msg("size(...)");
  return 0;
}

//-------------------
//===================

/// access to calibration constants
Query& AreaDetector::query(const event_t& evt) {
  return _query;
}

/// access to calibration constants
const NDArray<common_mode_t>& AreaDetector::common_mode(const event_t& evt) {
  return calib_pars()->common_mode(query(evt));
  //  _default_msg(std::string("common_mode(...)"));
  //  return _common_mode;
}

const NDArray<pedestals_t>& AreaDetector::pedestals(const event_t& evt) {
  return calib_pars()->pedestals(query(evt));
  //  _default_msg("pedestals(...)");
  //  return _pedestals;
}

const NDArray<pixel_rms_t>& AreaDetector::rms(const event_t& evt) {
  return calib_pars()->rms(query(evt));
  //  _default_msg("rms(...)");
  //  return _pixel_rms;
}

const NDArray<pixel_status_t>& AreaDetector::status(const event_t& evt) {
  return calib_pars()->status(query(evt));
  //  _default_msg("status(...)");
  //  return _pixel_status;
}

const NDArray<pixel_gain_t>& AreaDetector::gain(const event_t& evt) {
  return calib_pars()->gain(query(evt));
  //  _default_msg("gain(...)");
  //  return _pixel_gain;
}

const NDArray<pixel_offset_t>& AreaDetector::offset(const event_t& evt) {
  return calib_pars()->offset(query(evt));
  //  _default_msg("offset(...)");
  //  return _pixel_offset;
}

const NDArray<pixel_bkgd_t>& AreaDetector::background(const event_t& evt) {
  return calib_pars()->background(query(evt));
  //  _default_msg("background(...)");
  //  return _pixel_bkgd;
}

const NDArray<pixel_mask_t>& AreaDetector::mask_calib(const event_t& evt) {
  return calib_pars()->mask_calib(query(evt));
  //  _default_msg("mask_calib(...)");
  //  return _pixel_mask;
}

const NDArray<pixel_mask_t>& AreaDetector::mask_from_status(const event_t& evt) {
  return calib_pars()->mask_from_status(query(evt));
  //  _default_msg("mask_from_status(...)");
  //  return _pixel_mask;
}

const NDArray<pixel_mask_t>& AreaDetector::mask_edges(const event_t& evt, const size_t& nnbrs) {
  return calib_pars()->mask_edges(query(evt));
  //  _default_msg("mask_edges(...)");
  //  return _pixel_mask;
}

const NDArray<pixel_mask_t>& AreaDetector::mask_neighbors(const event_t& evt, const size_t& nrows, const size_t& ncols) {
  return calib_pars()->mask_neighbors(query(evt));
  //  _default_msg("mask_neighbors(...)");
  //  return _pixel_mask;
}

const NDArray<pixel_mask_t>& AreaDetector::mask_bits(const event_t& evt, const size_t& mbits) {
  return calib_pars()->mask_bits(query(evt));
  //  _default_msg("mask(...)");
  //  return _pixel_mask;
}

const NDArray<pixel_mask_t>& AreaDetector::mask(const event_t& evt, const bool& calib,
					                        const bool& sataus,
                                                                const bool& edges,
						                const bool& neighbors) {
  return calib_pars()->mask(query(evt));
  //  _default_msg("mask(...)");
  //  return _pixel_mask;
}

/// access to raw, calibrated data, and image


const NDArray<raw_t>& AreaDetector::raw(const event_t& evt) {
  _default_msg("raw(...)");
  return _raw;
}

const NDArray<calib_t>& AreaDetector::calib(const event_t& evt) {
  _default_msg("calib(...)");
  return _calib;
}

const NDArray<image_t>& AreaDetector::image(const event_t& evt) {
  _default_msg("image(...)");
  return _image;
}

const NDArray<image_t>& AreaDetector::image(const event_t& evt, const NDArray<image_t>& nda) {
  _default_msg("image(...)");
  return _image;
}

const NDArray<image_t>& AreaDetector::array_from_image(const event_t& evt, const NDArray<image_t>&) {
  _default_msg("array_from_image(...)");
  return _image;
}

void AreaDetector::move_geo(const event_t& evt, const pixel_size_t& dx,  const pixel_size_t& dy,  const pixel_size_t& dz) {
  _default_msg("move_geo(...)");
}

void AreaDetector::tilt_geo(const event_t& evt, const tilt_angle_t& dtx, const tilt_angle_t& dty, const tilt_angle_t& dtz) {
  _default_msg("tilt_geo(...)");
}

/// access to geometry
const geometry_t& AreaDetector::geometry(const event_t& evt) {
  return calib_pars()->geometry(query(evt));
  //  _default_msg("geometry(...)");
  //  return _geometry;
}

const NDArray<pixel_idx_t>& AreaDetector::indexes(const event_t& evt, const size_t& axis) {
  return calib_pars()->indexes(query(evt));
  //  _default_msg("indexes(...)");
  //  return _pixel_idx;
}

const NDArray<pixel_coord_t>& AreaDetector::coords(const event_t& evt, const size_t& axis) {
  return calib_pars()->coords(query(evt));
  //  _default_msg("coords(...)");
  //  return _pixel_coord;
}

const NDArray<pixel_size_t>& AreaDetector::pixel_size(const event_t& evt, const size_t& axis) {
  return calib_pars()->pixel_size(query(evt));
  //  _default_msg("pixel_size(...)");
  //  return _pixel_size;
}

const NDArray<pixel_size_t>& AreaDetector::image_xaxis(const event_t& evt) {
  return calib_pars()->image_xaxis(query(evt));
  //  _default_msg("image_xaxis(...)");
  //  return _pixel_size;
}

const NDArray<pixel_size_t>& AreaDetector::image_yaxis(const event_t& evt) {
  return calib_pars()->image_yaxis(query(evt));
  //  _default_msg("image_yaxis(...)");
  //  return _pixel_size;
}

calib::CalibPars* AreaDetector::calib_pars() {
  if(! _calib_pars) _calib_pars = calib::getCalibPars(detname().c_str());
  return _calib_pars;
}

calib::CalibPars* AreaDetector::calib_pars_updated() {
  if(_calib_pars) {delete _calib_pars; _calib_pars=0;}
  return calib_pars();
}

//-------------------
//-------------------

} // namespace detector

//-------------------
