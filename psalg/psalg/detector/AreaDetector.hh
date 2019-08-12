#ifndef PSALG_AREADETECTOR_H
#define PSALG_AREADETECTOR_H
//-----------------------------

#include <iostream>
#include <string>
#include <assert.h>

#include "psalg/calib/Query.hh"
#include "psalg/calib/CalibParsStore.hh" // CalibPars, getCalibPars
#include "psalg/calib/NDArray.hh" // NDArray
#include "psalg/calib/AreaDetectorTypes.hh" // event_t

#include "psalg/detector/Detector.hh"
#include "psalg/detector/UtilsConfig.hh" // configNames

#include "xtcdata/xtc/DataIter.hh"
#include "xtcdata/xtc/ConfigIter.hh"
typedef XtcData::ConfigIter ConfigIter;

using namespace std;
using namespace psalg;

namespace detector {

typedef unsigned index_t; // index of data in the xtc data types
typedef int64_t cfg_int64_t;

//-----------------------------

class AreaDetector : public Detector {
public:

  AreaDetector(const std::string& detname, ConfigIter& ci);
  AreaDetector(const std::string& detname);
  AreaDetector();

  virtual ~AreaDetector();

  void _default_msg(const std::string& msg=std::string()) const;

  virtual void set_indexes_config(XtcData::ConfigIter&);
  virtual void set_indexes_data(XtcData::DataIter&);

  //-------------------

  template <typename T>
  inline T config_value_for_index(XtcData::ConfigIter& ci, index_t i) {
    return ci.desc_shape().get_value<T>(i);
  }

  template <typename T>
  inline Array<T> config_array_for_index(XtcData::ConfigIter& ci, index_t i) {
    return ci.desc_shape().get_array<T>(i);
  }

  //-------------------

  inline DescData& descdata(XtcData::DataIter& di) {
    ConfigIter& ci = *_pconfig;
    NamesLookup& namesLookup = ci.namesLookup();
    return di.desc_value(namesLookup);
  }

  template <typename T>
  inline T data_value_for_index(XtcData::DescData& dd, index_t i) {
    return dd.get_value<T>(i);
  }

  template <typename T>
  inline T data_value_for_index(XtcData::DataIter& di, index_t i) {
    return descdata(di).get_value<T>(i);
    //return data_value_for_index<T>(descdata(di), i);
  }

  template <typename T>
  inline Array<T> data_array_for_index(XtcData::DescData& dd, index_t i) {
    return dd.get_array<T>(i);
  }

  template <typename T>
  inline Array<T> data_array_for_index(XtcData::DataIter& di, index_t i) {
    return descdata(di).get_array<T>(i);
    //return data_array_for_index<T>(descdata(di), i);
  }

  //-------------------

  //DEPRICATED
  virtual void process_config();
  virtual void process_data(XtcData::DataIter&);

  virtual void detid(std::ostream& os, const int ind=-1); //ind for panel, -1-for entire detector 
  virtual std::string detid(const int ind=-1);

  virtual const size_t ndim();
  virtual const size_t size();
  virtual shape_t* shape();

  virtual const size_t   ndim (const event_t&);
  virtual const size_t   size (const event_t&);
  virtual const shape_t* shape(const event_t&);

  virtual const void print_config();
  virtual const void print_data();
  virtual const void print_data(XtcData::DataIter&);

  template<typename T>
  void raw(XtcData::DescData& ddata, T*& pdata, const char* dataname="frame");

  template<typename T>
  void raw(XtcData::DataIter& datao, T*& pdata, const char* dataname="frame");

  template<typename T>
  void raw(XtcData::DescData& ddata, NDArray<T>& nda, const char* dataname="frame");

  template<typename T>
  void raw(XtcData::DataIter& datao, NDArray<T>& nda, const char* dataname="frame");

  /// access to calibration constants
  virtual const NDArray<common_mode_t>&   common_mode      (const event_t&);
  virtual const NDArray<pedestals_t>&     pedestals        (const event_t&);
  virtual const NDArray<double>&          pedestals_d      ();
  virtual const NDArray<pixel_rms_t>&     rms              (const event_t&);
  virtual const NDArray<pixel_status_t>&  status           (const event_t&);
  virtual const NDArray<pixel_gain_t>&    gain             (const event_t&);
  virtual const NDArray<pixel_offset_t>&  offset           (const event_t&);
  virtual const NDArray<pixel_bkgd_t>&    background       (const event_t&);
  virtual const NDArray<pixel_mask_t>&    mask_calib       (const event_t&);
  virtual const NDArray<pixel_mask_t>&    mask_from_status (const event_t&);
  virtual const NDArray<pixel_mask_t>&    mask_edges       (const event_t&, const size_t& nnbrs=8);
  virtual const NDArray<pixel_mask_t>&    mask_neighbors   (const event_t&, const size_t& nrows=1, const size_t& ncols=1);
  virtual const NDArray<pixel_mask_t>&    mask_bits        (const event_t&, const size_t& mbits=0177777);
  virtual const NDArray<pixel_mask_t>&    mask             (const event_t&, const bool& calib=true,
							                    const bool& sataus=true,
                                                                            const bool& edges=true,
							                    const bool& neighbors=true);

  /// access to raw, calibrated data, and image
  virtual const NDArray<raw_t>& raw(const event_t&);
  virtual const NDArray<calib_t>& calib(const event_t&);
  virtual const NDArray<image_t>& image(const event_t&);
  virtual const NDArray<image_t>& image(const event_t&, const NDArray<image_t>& nda);
  virtual const NDArray<image_t>& array_from_image(const event_t&, const NDArray<image_t>&);
  virtual void move_geo(const event_t&, const pixel_size_t& dx,  const pixel_size_t& dy,  const pixel_size_t& dz);
  virtual void tilt_geo(const event_t&, const tilt_angle_t& dtx, const tilt_angle_t& dty, const tilt_angle_t& dtz);

  /// access to geometry
  virtual const geometry_t& geometry(const event_t&);
  virtual NDArray<const pixel_coord_t>& coords     (const event_t&, const size_t& axis=0);
  virtual NDArray<const pixel_idx_t>&   indexes    (const event_t&, const size_t& axis=0);
  virtual NDArray<const pixel_size_t>&  pixel_size (const event_t&, const size_t& axis=0);
  virtual NDArray<const pixel_size_t>&  image_xaxis(const event_t&);
  virtual NDArray<const pixel_size_t>&  image_yaxis(const event_t&);

  calib::CalibPars* calib_pars();
  calib::CalibPars* calib_pars_updated();

  const std::string& expname()   {return _expname;}
  const std::string& calibtype() {return _calibtype;}
  const unsigned     runnum()    {return _runnum;}

  void set_expname(const std::string& expname) {_expname = expname;}
  void set_runnum(unsigned runnum) {_runnum = runnum;}
  void set_calibtype(const std::string& calibtype) {_calibtype = calibtype;}

  Query& query();
  Query& query(const event_t&);

  AreaDetector(const AreaDetector&) = delete;
  AreaDetector& operator = (const AreaDetector&) = delete;

  //---------------------------

  cfg_int64_t maxModulesPerDetector;
  cfg_int64_t numberOfModules;
  cfg_int64_t numberOfRows;
  cfg_int64_t numberOfColumns;
  cfg_int64_t numberOfPixels;

protected:
  shape_t*                _shape;
  ConfigIter*             _pconfig;
  int                     _ind_data;
  void _set_index_data(XtcData::DescData& ddata, const char* dataname);

private:

  calib::CalibPars*       _calib_pars;

  //std::string _detname;
  NDArray<raw_t>          _raw;
  NDArray<calib_t>        _calib;
  NDArray<image_t>        _image;

  std::string             _expname = "NOT_DEFINED";
  std::string             _calibtype = "pedestals";
  unsigned                _runnum = 0;
  Query                   _query;

  /*
  NDArray<common_mode_t>  _common_mode;
  NDArray<pedestals_t>    _pedestals;

  NDArray<pixel_rms_t>    _pixel_rms;
  NDArray<pixel_status_t> _pixel_status;
  NDArray<pixel_gain_t>   _pixel_gain;
  NDArray<pixel_offset_t> _pixel_offset;
  NDArray<pixel_bkgd_t>   _pixel_bkgd;
  NDArray<pixel_mask_t>   _pixel_mask;

  geometry_t              _geometry;
  NDArray<pixel_idx_t>    _pixel_idx;
  NDArray<pixel_coord_t>  _pixel_coord;
  NDArray<pixel_size_t>   _pixel_size;
  */

}; // class

} // namespace detector

#endif // PSALG_AREADETECTOR_H
//-----------------------------
