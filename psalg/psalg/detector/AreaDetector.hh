/**
 * @file
 * @brief Declares detector::AreaDetector, the base class for area detectors, plus the index_t and cfg_int64_t aliases.
 */
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
/** Global alias for XtcData::ConfigIter. */
typedef XtcData::ConfigIter ConfigIter;

using namespace std;
using namespace psalg;

namespace detector {

/** uint64_t; the trailing comment says it is an index of data in the xtc data types. */
typedef uint64_t index_t; // index of data in the xtc data types
/** int64_t; marked DEPRICATED in the trailing comment. */
typedef int64_t cfg_int64_t; // DEPRICATED

//-----------------------------

/**
 * Base class for area detectors. Config and data fields are read through a stored ConfigIter; calibration constants and geometry are forwarded to a CalibPars object created on first use with calib::getCalibPars(detname).
 * Many virtual methods are defaults that only log a warning asking for re-implementation in a derived class.
 */
class AreaDetector : public Detector {
public:

  /** Construct for detname (type AREA_DETECTOR) and keep a pointer to ci for later config and data access. */
  AreaDetector(const std::string& detname, ConfigIter& ci);
  /** Construct for detname (type AREA_DETECTOR) with no ConfigIter. */
  AreaDetector(const std::string& detname);
  /** Default constructor: Detector() (name "NoDevice"), no ConfigIter. */
  AreaDetector();

  /** Delete the CalibPars object and the shape array if they were created (the array is allocated with new[] but released with delete). */
  virtual ~AreaDetector();

  /** Log a WARNING that AreaDetector::msg is a default method that should be re-implemented in the derived class. */
  void _default_msg(const std::string& msg=std::string()) const;

  /** Default: only logs the _default_msg() warning. */
  virtual void _set_indexes_config(XtcData::ConfigIter&);
  /** Default: only logs the _default_msg() warning. */
  virtual void _set_indexes_data(XtcData::DataIter&);

  //-------------------

  /** Return field i of ci.desc_shape() as a scalar of type T (DescData::get_value<T>()). */
  template <typename T>
  inline T config_value_for_index(XtcData::ConfigIter& ci, index_t i) {
    return ci.desc_shape().get_value<T>(i);
  }

  /** Return field i of ci.desc_shape() as an Array<T> (DescData::get_array<T>()). */
  template <typename T>
  inline Array<T> config_array_for_index(XtcData::ConfigIter& ci, index_t i) {
    return ci.desc_shape().get_array<T>(i);
  }

  /** Return field i of the stored ConfigIter's desc_shape() as a scalar of type T; asserts that a ConfigIter is stored. */
  template <typename T>
  inline T config_value_for_index(index_t i) {
    assert(_pconfit != 0);
    return _pconfit->desc_shape().get_value<T>(i);
  }

  /** Return field i of the stored ConfigIter's desc_shape() as an Array<T>; asserts that a ConfigIter is stored. */
  template <typename T>
  inline Array<T> config_array_for_index(index_t i) {
    assert(_pconfit != 0);
    return _pconfit->desc_shape().get_array<T>(i);
  }

  //-------------------

  /** Return di.desc_value() built with the NamesLookup of the stored ConfigIter (not checked to be set). */
  inline DescData& descdata(XtcData::DataIter& di) {
    ConfigIter& ci = *_pconfit;
    NamesLookup& namesLookup = ci.namesLookup();
    return di.desc_value(namesLookup);
  }

  /** Return dd.get_value<T>(i). */
  template <typename T>
  inline T data_value_for_index(XtcData::DescData& dd, index_t i) {
    return dd.get_value<T>(i);
  }

  /** Return descdata(di).get_value<T>(i). */
  template <typename T>
  inline T data_value_for_index(XtcData::DataIter& di, index_t i) {
    return descdata(di).get_value<T>(i);
    //return data_value_for_index<T>(descdata(di), i);
  }

  /** Return dd.get_array<T>(i). */
  template <typename T>
  inline Array<T> data_array_for_index(XtcData::DescData& dd, index_t i) {
    return dd.get_array<T>(i);
  }

  /** Return descdata(di).get_array<T>(i). */
  template <typename T>
  inline Array<T> data_array_for_index(XtcData::DataIter& di, index_t i) {
    return descdata(di).get_array<T>(i);
    //return data_array_for_index<T>(descdata(di), i);
  }

  //-------------------

  //DEPRICATED
  /**
   * Print to stdout the Names of the stored ConfigIter's shape() entry and, for each field, its name, rank, type and element size, plus the value for INT64 scalars or the elements of arrays read as uint32_t.
   * Marked DEPRICATED in the header.
   */
  virtual void process_config();
  /**
   * Log the _default_msg() warning, then print each data field's name, rank and type, with the value for INT64 scalars or the first five elements of arrays read as uint16_t.
   * Marked DEPRICATED in the header.
   */
  virtual void process_data(XtcData::DataIter&);

  /** Write "default_area_detector_id" to os; ind is ignored. */
  virtual void detid(std::ostream& os, const int ind=-1); //ind for panel, -1-for entire detector 
  /** Return what detid(std::ostream&, ind) writes, as a string. */
  virtual std::string detid(const int ind=-1);

  /** Return 3 if numberOfModules > 1, else 2. */
  virtual const size_t ndim();
  /** Return numberOfPixels. */
  virtual const size_t size();
  /**
   * Return the shape array, allocated on the first call from {numberOfModules, numberOfRows, numberOfColumns} if numberOfModules > 1, else {numberOfRows, numberOfColumns}.
   * Later calls return the same array.
   */
  virtual shape_t* shape();

  /** Default: logs the _default_msg() warning and returns 0. */
  virtual const size_t   ndim (const event_t&);
  /** Default: logs the _default_msg() warning and returns 0. */
  virtual const size_t   size (const event_t&);
  /** Default: logs the _default_msg() warning; if no shape array exists yet, allocates {1, 2, 3}; returns the shape array. */
  virtual const shape_t* shape(const event_t&);

  /** Default: only logs the _default_msg() warning. */
  virtual const void print_config();
  /** Default: only logs the _default_msg() warning. */
  virtual const void print_data();
  /** Default: only logs the _default_msg() warning. */
  virtual const void print_data(XtcData::DataIter&);

  /**
   * Set pdata to the data of the array field named dataname in ddata. The field index is looked up on the first call only and reused afterwards; if no field matches it stays -1 (not checked).
   * Defined in AreaDetector.cc and instantiated there for uint16_t.
   */
  template<typename T>
  void raw(XtcData::DescData& ddata, T*& pdata, const char* dataname="frame");

  /** Same as raw(DescData&, T*&, const char*), using datao.desc_value() with the stored ConfigIter's NamesLookup. Instantiated for uint16_t. */
  template<typename T>
  void raw(XtcData::DataIter& datao, T*& pdata, const char* dataname="frame");

  /** Like raw(DescData&, T*&, const char*), then set the shape of nda from shape() and ndim() and point nda at the data without copying (set_data_buffer()). Instantiated for uint16_t. */
  template<typename T>
  void raw(XtcData::DescData& ddata, NDArray<T>& nda, const char* dataname="frame");

  /** Same as raw(DescData&, NDArray<T>&, const char*), using datao.desc_value() with the stored ConfigIter's NamesLookup. Instantiated for uint16_t. */
  template<typename T>
  void raw(XtcData::DataIter& datao, NDArray<T>& nda, const char* dataname="frame");

  /// access to calibration constants
  virtual const NDArray<common_mode_t>&   common_mode      (const event_t&);
  /** Return calib_pars()->pedestals(query(evt)); query(evt) returns the stored Query without refreshing it. */
  virtual const NDArray<pedestals_t>&     pedestals        (const event_t&);
  /** Return calib_pars()->pedestals_d(query()); query() first refreshes the Query from detname, expname, calibtype and runnum. */
  virtual const NDArray<double>&          pedestals_d      ();
  /** Return calib_pars()->rms(query(evt)). */
  virtual const NDArray<pixel_rms_t>&     rms              (const event_t&);
  /** Return calib_pars()->status(query(evt)). */
  virtual const NDArray<pixel_status_t>&  status           (const event_t&);
  /** Return calib_pars()->gain(query(evt)). */
  virtual const NDArray<pixel_gain_t>&    gain             (const event_t&);
  /** Return calib_pars()->offset(query(evt)). */
  virtual const NDArray<pixel_offset_t>&  offset           (const event_t&);
  /** Return calib_pars()->background(query(evt)). */
  virtual const NDArray<pixel_bkgd_t>&    background       (const event_t&);
  /** Return calib_pars()->mask_calib(query(evt)). */
  virtual const NDArray<pixel_mask_t>&    mask_calib       (const event_t&);
  /** Return calib_pars()->mask_from_status(query(evt)). */
  virtual const NDArray<pixel_mask_t>&    mask_from_status (const event_t&);
  /** Return calib_pars()->mask_edges(query(evt)); nnbrs is ignored. */
  virtual const NDArray<pixel_mask_t>&    mask_edges       (const event_t&, const size_t& nnbrs=8);
  /** Return calib_pars()->mask_neighbors(query(evt)); nrows and ncols are ignored. */
  virtual const NDArray<pixel_mask_t>&    mask_neighbors   (const event_t&, const size_t& nrows=1, const size_t& ncols=1);
  /** Return calib_pars()->mask_bits(query(evt)); mbits is ignored. */
  virtual const NDArray<pixel_mask_t>&    mask_bits        (const event_t&, const size_t& mbits=0177777);
  /** Return calib_pars()->mask(query(evt)); the bool flags are ignored. */
  virtual const NDArray<pixel_mask_t>&    mask             (const event_t&, const bool& calib=true,
							                    const bool& sataus=true,
                                                                            const bool& edges=true,
							                    const bool& neighbors=true);

  /// access to raw, calibrated data, and image
  virtual void load_calib_constants();

  /** Default: logs the _default_msg() warning and returns a private array that this class never fills. */
  virtual const NDArray<raw_t>& raw(const event_t&);
  /** Default: logs the _default_msg() warning and returns a private array that this class never fills. */
  virtual const NDArray<calib_t>& calib(const event_t&);
  /** Default: logs the _default_msg() warning and returns a private array that this class never fills. */
  virtual const NDArray<image_t>& image(const event_t&);
  /** Default: logs the _default_msg() warning and returns the private image array; nda is not used. */
  virtual const NDArray<image_t>& image(const event_t&, const NDArray<image_t>& nda);
  /** Default: logs the _default_msg() warning and returns the private image array; the input array is not used. */
  virtual const NDArray<image_t>& array_from_image(const event_t&, const NDArray<image_t>&);
  /** Default: only logs the _default_msg() warning. */
  virtual void move_geo(const event_t&, const pixel_size_t& dx,  const pixel_size_t& dy,  const pixel_size_t& dz);
  /** Default: only logs the _default_msg() warning. */
  virtual void tilt_geo(const event_t&, const tilt_angle_t& dtx, const tilt_angle_t& dty, const tilt_angle_t& dtz);

  /// access to geometry
  virtual const geometry_t& geometry(const event_t&);
  /** Return calib_pars()->coords(query(evt)); axis is ignored (CalibPars::coords() takes the axis from the Query's AXISNUM). */
  virtual NDArray<const pixel_coord_t>& coords     (const event_t&, const size_t& axis=0);
  /** Return calib_pars()->indexes(query(evt)); axis is ignored (CalibPars::indexes() takes the axis from the Query's AXISNUM). */
  virtual NDArray<const pixel_idx_t>&   indexes    (const event_t&, const size_t& axis=0);
  /** Return calib_pars()->pixel_size(query(evt)); axis is ignored. */
  virtual NDArray<const pixel_size_t>&  pixel_size (const event_t&, const size_t& axis=0);
  /** Return calib_pars()->image_xaxis(query(evt)). */
  virtual NDArray<const pixel_size_t>&  image_xaxis(const event_t&);
  /** Return calib_pars()->image_yaxis(query(evt)). */
  virtual NDArray<const pixel_size_t>&  image_yaxis(const event_t&);

  /** Return the CalibPars object, creating it on first use with calib::getCalibPars(detname()), which throws for detector types it does not support. */
  calib::CalibPars* calib_pars();
  /** Delete the current CalibPars object (if any) and return a new one from calib_pars(). */
  calib::CalibPars* calib_pars_updated();

  /** Return the experiment name (initially "NOT_DEFINED"). */
  const std::string& expname()   {return _expname;}
  /** Return the calibration type string (initially "pedestals"). */
  const std::string& calibtype() {return _calibtype;}
  /** Return the run number (initially 0). */
  const unsigned     runnum()    {return _runnum;}

  /** Set the experiment name used by query(). */
  void set_expname(const std::string& expname) {_expname = expname;}
  /** Set the run number used by query(). */
  void set_runnum(unsigned runnum) {_runnum = runnum;}
  /** Set the calibration type string used by query(). */
  void set_calibtype(const std::string& calibtype) {_calibtype = calibtype;}

  /** Refresh the stored Query with set_paremeters(detname, expname, calibtype, runnum) and return it. */
  Query& query();
  /** Return the stored Query unchanged; evt is ignored. */
  Query& query(const event_t&);

  /** Copy construction is disabled. */
  AreaDetector(const AreaDetector&) = delete;
  /** Copy assignment is disabled. */
  AreaDetector& operator = (const AreaDetector&) = delete;

  //---------------------------

  int64_t maxModulesPerDetector;  ///< Not initialized by AreaDetector; set by subclasses (e.g. from the config field "MaxModulesPerDetector" in AreaDetectorCspad).
  int64_t numberOfModules;  ///< Not initialized by AreaDetector; set by subclasses. Used by ndim() and shape().
  int64_t numberOfRows;  ///< Not initialized by AreaDetector; set by subclasses. Used by shape().
  int64_t numberOfColumns;  ///< Not initialized by AreaDetector; set by subclasses. Used by shape().
  int64_t numberOfPixels;  ///< Not initialized by AreaDetector; set by subclasses. Returned by size().

protected:
  shape_t*                _shape;
  ConfigIter*             _pconfit = 0;
  DataIter*               _pdatait = 0;
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
