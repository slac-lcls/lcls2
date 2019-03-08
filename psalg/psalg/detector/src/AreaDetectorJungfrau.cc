
//#include <string>
//#include <iostream> // for cout, puts
#include <iomanip>    // for setw, hex, setprecision, right
#include <sstream>    // for stringstream

#include "psalg/detector/AreaDetectorJungfrau.hh"
#include "psalg/utils/Logger.hh" // for MSG

using namespace std;
using namespace psalg;

namespace detector {

//-----------------------------

AreaDetectorJungfrau::AreaDetectorJungfrau(const std::string& detname, XtcData::ConfigIter& configo)
  : AreaDetector(detname, configo) {
  MSG(DEBUG, "In c-tor AreaDetectorJungfrau(detname, configo) for " << detname);
  process_config();
}

AreaDetectorJungfrau::AreaDetectorJungfrau(const std::string& detname)
  : AreaDetector(detname) {
  MSG(DEBUG, "In c-tor AreaDetectorJungfrau(detname) for " << detname);
}

AreaDetectorJungfrau::~AreaDetectorJungfrau() {
  MSG(DEBUG, "In d-tor AreaDetectorJungfrau for " << detname());
}

void AreaDetectorJungfrau::_class_msg(const std::string& msg) {
  MSG(INFO, "In AreaDetectorJungfrau::"<< msg);
}

  /*
const shape_t* AreaDetectorJungfrau::shape(const event_t&) {
  _class_msg("shape(...)");
  return &AreaDetector::_shape[0];
  //return &_shape[0];
}

const size_t AreaDetectorJungfrau::ndim(const event_t&) {
  _class_msg("ndim(...)");
  return 0;
}

const size_t AreaDetectorJungfrau::size(const event_t&) {
  _class_msg("size(...)");
  return 123;
}

const shape_t* AreaDetectorJungfrau::shape() {
  _class_msg("shape(...)");
  return &AreaDetector::_shape[0];
}

const size_t AreaDetectorJungfrau::size() {
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

void AreaDetectorJungfrau::process_config() {

  XtcData::ConfigIter& configo = *_pconfig;
  //XtcData::NamesId& namesId = configo.shape().namesId();
  XtcData::Names& names = configNames(configo);

  MSG(DEBUG, str_config_names(configo));

  //DESC_SHAPE(desc_shape, configo, namesLookup);
  XtcData::DescData& desc_shape = configo.desc_shape();

  //DESC_VALUE(desc_value, configo, namesLookup);
  //XtcData::DescData& desc_value = configo.desc_value();

  for (unsigned i = 0; i < names.num(); i++) {
      XtcData::Name& name = names.get(i);
      //XtcData::Name::DataType itype = name.type();
      const char* cname = name.name();
      //printf("%02d name: %-32s rank: %d type: %d el.size %02d",
      //       i, cname, name.rank(), itype, Name::get_element_size(itype));

      if (strcmp(cname, "MaxModulesPerDetector")    == 0) {maxModulesPerDetector = desc_shape.get_value<int64_t>(i); continue;}
      if (strcmp(cname, "numberOfModules")          == 0) {numberOfModules       = desc_shape.get_value<int64_t>(i); continue;}
      if (strcmp(cname, "MaxRowsPerModule")         == 0) {numberOfRows          = desc_shape.get_value<int64_t>(i); continue;}
      if (strcmp(cname, "numberOfColumnsPerModule") == 0) {numberOfColumns       = desc_shape.get_value<int64_t>(i); continue;}
      if (strcmp(cname, "numPixels")                == 0) {numberOfPixels        = desc_shape.get_value<int64_t>(i); continue;}

      for (unsigned m=0; m < MAX_NUMBER_OF_MODULES; m++) {
        char cbuf1[32]; sprintf(&cbuf1[0], "moduleConfig%d_firmwareVersion", m);
        char cbuf2[32]; sprintf(&cbuf2[0], "moduleConfig%d_moduleVersion", m);
        char cbuf3[32]; sprintf(&cbuf3[0], "moduleConfig%d_serialNumber", m);
        if (strcmp(cname, cbuf1) == 0) {firmwareVersion[m] = desc_shape.get_value<int64_t>(i); break;}
        if (strcmp(cname, cbuf2) == 0) {moduleVersion  [m] = desc_shape.get_value<int64_t>(i); break;}
        if (strcmp(cname, cbuf3) == 0) {serialNumber   [m] = desc_shape.get_value<int64_t>(i); break;}
      }

      /*
        if (name.type()==Name::INT64 and name.rank()==0)
  	     printf(" value: %ld\n", desc_shape.get_value<int64_t>(cname));

        if (strcmp(cname, "moduleConfig_shape") == 0 and name.rank()==1) {
            auto array = desc_shape.get_array<uint32_t>(i);
            for (unsigned k=0; k<array.shape()[0]; k++) { // for a rank 1 array
              cout << " ===> save: element " << k << " has value " << array(k);
            }
            cout << "\n";
        }
      */
  }
}

//-------------------

const void AreaDetectorJungfrau::print_config() {

  std::cout << "\n\n==== Attributes of configuration ====\n";
  std::cout << "detname                       : " << this->detname() << '\n';
  std::cout << "dettype                       : " << this->dettype() << '\n';
  std::cout << "maxModulesPerDetector         : " << this->maxModulesPerDetector << '\n';
  std::cout << "numberOfModules               : " << this->numberOfModules << '\n';
  std::cout << "numPixels                     : " << this->numberOfPixels << '\n';
  std::cout << "numberOfRows                  : " << this->numberOfRows << '\n';
  std::cout << "numberOfColumns               : " << this->numberOfColumns << '\n';

  std::cout << "==== Derived values ====\n";
  std::cout << "panel_id 0                    : " << AreaDetector::detid(0) << '\n' 
            << "panel_id 1                    : " << AreaDetector::detid(1) << '\n'
            << "detid                         : " << AreaDetector::detid() << '\n';
  std::cout << "ndim()                        : " << this->ndim() << '\n';
  std::cout << "size()                        : " << this->size() << '\n';

  const shape_t* pshape = AreaDetector::shape();
  std::cout << "shape()                       : (";
  for(unsigned i=0; i<this->ndim(); i++) {
    std::cout << pshape[i] << ((i<this->ndim()-1) ? ", " : ")\n");
  }
}

//-------------------

void AreaDetectorJungfrau::process_data(XtcData::DataIter& datao) {

    ConfigIter& configo = *_pconfig;
    NamesLookup& namesLookup = configo.namesLookup();
    DescData& descdata = datao.desc_value(namesLookup);
    Names& names = descdata.nameindex().names();

    printf("\n------ %d Names and values for data ---------\n", names.num());
    for (unsigned i = 0; i < names.num(); i++) {
        Name& name = names.get(i);
        printf("%02d name: %-32s rank: %d type: %d", i, name.name(), name.rank(), name.type());
        if (name.type()==Name::INT64 and name.rank()==0) {
	  //printf(" value %ld\n", descdata.get_value<int64_t>(name.name()));
	  printf(" value %ld\n", descdata.get_value<int64_t>(i));
        }
	else if (name.type() == 1) {
	  uint16_t *data = descdata.get_array<uint16_t>(i).data();
	  printf("  ===> %d %d %d %d %d\n", data[0],data[1],data[2],data[3],data[4]);

	  // cpo - remove comments if needed
	  //uint16_t *data0 = descdata.get_array<uint16_t>(0).data();
	  //for (int k=0; k<120; k++) printf("     k: %03d  data_v: %d\n", k, data0[k]);
	}
	else if (name.rank() == 1 && name.type() == 7) {
	  int32_t *data = descdata.get_array<int32_t>(i).data();
	  printf("  ===> %d %d %d\n", data[0],data[1],data[2]);
	  /*
	  uint32_t *sh = descdata.get_array<uint32_t>(0).data();
          cout << " ==> save: " << sh[0] << '\n';
	  for (int k=0; k<120; k++) // cout << "     k: " << k << "  v: " << shape_cfg[k] << '\n';
	  	    printf("     k: %03d  v: %d\n", k, sh[k]);
	  */
	}
	else {
          printf("  ==> TBD\n");
	}
    }
}

//-------------------

void AreaDetectorJungfrau::_panel_id(std::ostream& os, const int ind) {
  os << std::hex << moduleVersion[ind] << '-' 
     << std::hex << firmwareVersion[ind] << '-' 
     << std::hex << setfill('0')  << serialNumber[ind];
  // << std::hex << std::setw(10) << std::setprecision(8) << std::right<< setfill('0')
}

//-------------------

void AreaDetectorJungfrau::detid(std::ostream& os, const int ind) {
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

NDArray<rawjf_t>& AreaDetectorJungfrau::raw(XtcData::DescData& ddata) {
    if(_ind_data < 0) _set_index_data(ddata, "frame");
    rawjf_t* pdata = ddata.get_array<rawjf_t>(_ind_data).data();
    _raw.set_shape(shape(), ndim());
    _raw.set_data_buffer(pdata);
    return _raw;
}

//-------------------

NDArray<rawjf_t>& AreaDetectorJungfrau::raw(XtcData::DataIter& datao) {
    DescData&  ddata = datao.desc_value(_pconfig->namesLookup());
    return raw(ddata);
}

//-------------------

/* implemented in AreaDetector
std::string AreaDetectorJungfrau::detid(const int& ind) {
  std::stringstream ss;
  detid(ss, ind);
  return ss.str();
}
*/

//-------------------
//-------------------

} // namespace detector

//-----------------------------
