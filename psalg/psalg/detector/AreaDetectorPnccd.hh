/**
 * @file
 * @brief Declares detector::AreaDetectorPnccd, the AreaDetector subclass for pnCCD config and data fields, and its element-type aliases.
 */
#ifndef PSALG_AREADETECTORPNCCD_H
#define PSALG_AREADETECTORPNCCD_H
//-----------------------------

#include <stdint.h>  // uint8_t, uint32_t, etc.
#include "psalg/detector/AreaDetector.hh"

//using namespace std;
using namespace psalg;

namespace detector {

/** uint16_t; element type of AreaDetectorPnccd::raw() (trailing comment: raw type of pnccd data). */
typedef uint16_t pnccd_raw_t;       // raw   type of pnccd data
/** float; element type of AreaDetectorPnccd::calib() (trailing comment: calib type of pnccd data). */
typedef float    pnccd_calib_t;     // calib type of pnccd data
/** double; element type of the pedestals that calib() subtracts (trailing comment: pedestals type of pnccd data). */
typedef double   pnccd_pedestals_t; // pedestals type of pnccd data

//-----------------------------
/** AreaDetector subclass for pnCCD: finds the indexes of the config fields and of the per-module data fields (frame0_... to frame3_...) by name, gives typed accessors for them, and builds raw() and pedestal-subtracted calib() arrays from the module data. */
class AreaDetectorPnccd : public AreaDetector {
public:

  //------------------------------

  /** Construct with AreaDetector(detname, configiter). Unlike AreaDetectorOpal it does not call _set_indexes_config(). */
  AreaDetectorPnccd(const std::string& detname, XtcData::ConfigIter& configiter);
  /** Construct with AreaDetector(detname); no ConfigIter is stored. */
  AreaDetectorPnccd(const std::string& detname);
  /** Default constructor (AreaDetector()). */
  AreaDetectorPnccd();
  /** Destructor; only logs a debug message. */
  virtual ~AreaDetectorPnccd();

  /**
   * Write ind zero-padded to 4 characters; if ind is negative, write the ids of modules 0 .. numberOfModules-1 joined by '_' (e.g. 0000_0001).
   * Asserts ind < MAX_NUMBER_OF_MODULES and 0 < numberOfModules <= MAX_NUMBER_OF_MODULES.
   */
  virtual void detid(std::ostream& os, const int ind=-1); //ind for panel, -1-for entire detector

  /**
   * Store configiter, build the per-module data field names, and record the index of each known config field (Version, TypeId, numLinks, ..., timingFName_shape) by name.
   * Then set maxModulesPerDetector = MAX_NUMBER_OF_MODULES, numberOfModules = numSubmodules(), numberOfRows = numSubmoduleRows(), numberOfColumns = numSubmoduleChannels() and numberOfPixels to their product.
   */
  virtual void _set_indexes_config(XtcData::ConfigIter&);
  /**
   * Print each data field (index, name, rank, type) to stdout and record the indexes of Version, TypeId, numLinks, frame_shape and, per module, frameM_frameNumber, _timeStampHi, _timeStampLo, _specialWord, _data and __data.
   * Needs _set_indexes_config() first. The numLinks data index overwrites the config index that numLinks() uses.
   */
  virtual void _set_indexes_data(XtcData::DataIter&);

  /** Print the recorded config field indexes to stdout. */
  virtual const void print_config_indexes();
  /** Print the recorded data field indexes to stdout (the Version line shows the config index of Version). */
  virtual const void print_data_indexes();
  /** Print detname, dettype, all config values, the derived counts, detid(), ndim() and size() to stdout. */
  virtual const void print_config();
  /** Print the data values of di and, for each module, frame number, time stamps, special word and both data arrays to stdout. Does not call _set_indexes_data(). */
  virtual const void print_data(XtcData::DataIter&);

  /** Log an INFO message "In AreaDetectorPnccd::" followed by msg. */
  void _class_msg(const std::string& msg=std::string());

  /** Copy the 1-d data array (frameM__data) of each of the MAX_NUMBER_OF_MODULES modules into the internal raw array, shaped by shape() and ndim(), and return it. */
  virtual NDArray<pnccd_raw_t>& raw(XtcData::DescData&);
  /** Return raw(descdata(di)); unlike AreaDetectorOpal it does not call _set_indexes_data(). */
  virtual NDArray<pnccd_raw_t>& raw(XtcData::DataIter& di) {return raw(descdata(di));}

  /** Forward to AreaDetector::calib(evt), a default that logs a warning; declared here to avoid a hidden-overload warning. */
  virtual const NDArray<calib_t>& calib(const event_t& evt) {return AreaDetector::calib(evt);} // Silence compiler "hidden" warning
  /**
   * Build the raw array as raw(dd) does, then fill and return an internal float array with raw minus pedestals, element by element.
   * The pedestals pointer is set only by load_calib_constants(); it is not checked.
   */
  virtual NDArray<pnccd_calib_t>& calib(XtcData::DescData&);
  /** Return calib(descdata(di)). */
  virtual NDArray<pnccd_calib_t>& calib(XtcData::DataIter& di) {return calib(descdata(di));}

  /** Set the calib type to "pedestals", get pedestals_d() (double constants from the CalibPars backend), print them to stdout and keep a pointer to them for calib(). */
  virtual void load_calib_constants();

  // implemented in AreaDetector
  /// shape, size, ndim of data from configuration object
  //virtual const size_t   ndim (); // defiled in superclass AreaDetector
  //virtual const size_t   size ();
  //virtual const shape_t* shape();

  /// access to calibration constants
  /*
  const NDArray<common_mode_t>&   common_mode      (const event_t&);
  const NDArray<pedestals_t>&     pedestals        (const event_t&);
  const NDArray<pixel_rms_t>&     rms              (const event_t&);
  const NDArray<pixel_status_t>&  status           (const event_t&);
  const NDArray<pixel_gain_t>&    gain             (const event_t&);
  const NDArray<pixel_offset_t>&  offset           (const event_t&);
  const NDArray<pixel_bkgd_t>&    background       (const event_t&);
  const NDArray<pixel_mask_t>&    mask_calib       (const event_t&);
  const NDArray<pixel_mask_t>&    mask_from_status (const event_t&);
  const NDArray<pixel_mask_t>&    mask_edges       (const event_t&, const size_t& nnbrs=8);
  const NDArray<pixel_mask_t>&    mask_neighbors   (const event_t&, const size_t& nrows=1, const size_t& ncols=1);
  const NDArray<pixel_mask_t>&    mask             (const event_t&, const size_t& mbits=0177777);
  const NDArray<pixel_mask_t>&    mask             (const event_t&, const bool& calib=true,
  						                    const bool& sataus=true,
                                                                    const bool& edges=true,
  						                    const bool& neighbors=true);

  /// access to raw, calibrated data, and image
  const NDArray<raw_t>&   raw  (const event_t&);
  const NDArray<calib_t>& calib(const event_t&);
  const NDArray<image_t>& image(const event_t&);
  const NDArray<image_t>& image(const event_t&, const NDArray<image_t>& nda);
  const NDArray<image_t>& array_from_image(const event_t&, const NDArray<image_t>&);
  void move_geo(const event_t&, const pixel_size_t& dx,  const pixel_size_t& dy,  const pixel_size_t& dz);
  void tilt_geo(const event_t&, const tilt_angle_t& dtx, const tilt_angle_t& dty, const tilt_angle_t& dtz);

  /// access to geometry
  const geometry_t* geometry(const event_t&);
  const NDArray<pixel_idx_t>&   indexes    (const event_t&, const size_t& axis=0);
  const NDArray<pixel_coord_t>& coords     (const event_t&, const size_t& axis=0);
  const NDArray<pixel_size_t>&  pixel_size (const event_t&, const size_t& axis=0);
  const NDArray<pixel_size_t>&  image_xaxis(const event_t&);
  const NDArray<pixel_size_t>&  image_yaxis(const event_t&);
  */

  enum {MAX_NUMBER_OF_MODULES=4 /**< Value 4: size of the per-module index arrays; raw() always copies this many modules. */ };

  // convenience methods of AreaDetectorPnccd ONLY!
  /** Return config field "Version" of the stored ConfigIter as cfg_int64_t, at the recorded index (0 if not found); aborts if the field is not INT64. */
  inline cfg_int64_t Version             () {return config_value_for_index<cfg_int64_t>(*_pconfit, _Version);}
  /** Return config field "TypeId" of the stored ConfigIter as cfg_int64_t, at the recorded index (0 if not found); aborts if the field is not INT64. */
  inline cfg_int64_t TypeId              () {return config_value_for_index<cfg_int64_t>(*_pconfit, _TypeId);}
  /** Return config field "numLinks" of the stored ConfigIter as cfg_int64_t, at the recorded index (0 if not found); aborts if the field is not INT64. After _set_indexes_data() it uses the data index of numLinks instead. */
  inline cfg_int64_t numLinks            () {return config_value_for_index<cfg_int64_t>(*_pconfit, _numLinks);}
  /** Return config field "numChannels" of the stored ConfigIter as cfg_int64_t, at the recorded index (0 if not found); aborts if the field is not INT64. */
  inline cfg_int64_t numChannels         () {return config_value_for_index<cfg_int64_t>(*_pconfit, _numChannels);}
  /** Return config field "camexMagic" of the stored ConfigIter as cfg_int64_t, at the recorded index (0 if not found); aborts if the field is not INT64. */
  inline cfg_int64_t camexMagic          () {return config_value_for_index<cfg_int64_t>(*_pconfit, _camexMagic);}
  /** Return config field "numRows" of the stored ConfigIter as cfg_int64_t, at the recorded index (0 if not found); aborts if the field is not INT64. */
  inline cfg_int64_t numRows             () {return config_value_for_index<cfg_int64_t>(*_pconfit, _numRows);}
  /** Return config field "numSubmoduleRows" of the stored ConfigIter as cfg_int64_t, at the recorded index (0 if not found); aborts if the field is not INT64. */
  inline cfg_int64_t numSubmoduleRows    () {return config_value_for_index<cfg_int64_t>(*_pconfit, _numSubmoduleRows);}
  /** Return config field "payloadSizePerLink" of the stored ConfigIter as cfg_int64_t, at the recorded index (0 if not found); aborts if the field is not INT64. */
  inline cfg_int64_t payloadSizePerLink  () {return config_value_for_index<cfg_int64_t>(*_pconfit, _payloadSizePerLink);}
  /** Return config field "numSubmoduleChannels" of the stored ConfigIter as cfg_int64_t, at the recorded index (0 if not found); aborts if the field is not INT64. */
  inline cfg_int64_t numSubmoduleChannels() {return config_value_for_index<cfg_int64_t>(*_pconfit, _numSubmoduleChannels);}
  /** Return config field "numSubmodules" of the stored ConfigIter as cfg_int64_t, at the recorded index (0 if not found); aborts if the field is not INT64. */
  inline cfg_int64_t numSubmodules       () {return config_value_for_index<cfg_int64_t>(*_pconfit, _numSubmodules);}

  /** Return config field "info_shape" of the stored ConfigIter as Array<cfg_int64_t>, at the recorded index (0 if not found). */
  inline Array<cfg_int64_t> info_shape()        {return config_array_for_index<cfg_int64_t>(*_pconfit, _info_shape);}
  /** Return config field "timingFName_shape" of the stored ConfigIter as Array<cfg_int64_t>, at the recorded index (0 if not found). */
  inline Array<cfg_int64_t> timingFName_shape() {return config_array_for_index<cfg_int64_t>(*_pconfit, _timingFName_shape);}

  //data values, arrays
  /** Return data field "Version" of di as cfg_int64_t, at the index recorded by _set_indexes_data() (0 if not found); aborts if the field is not INT64. */
  inline cfg_int64_t data_Version (XtcData::DataIter& di) {return data_value_for_index<cfg_int64_t>(di, _data_Version);}
  /** Return data field "TypeId" of di as cfg_int64_t, at the index recorded by _set_indexes_data() (0 if not found); aborts if the field is not INT64. */
  inline cfg_int64_t data_TypeId  (XtcData::DataIter& di) {return data_value_for_index<cfg_int64_t>(di, _data_TypeId);}
  /** Return field numLinks of di as cfg_int64_t, using the shared numLinks index (config or data, whichever was recorded last); aborts if the field is not INT64. */
  inline cfg_int64_t data_numLinks(XtcData::DataIter& di) {return data_value_for_index<cfg_int64_t>(di, _numLinks);}
  /** Return data field frameM_frameNumber of module m in di as cfg_int64_t. m is not range-checked and the index is uninitialized until _set_indexes_data() finds the field. */
  inline cfg_int64_t frameNumber(XtcData::DataIter& di, unsigned m) {return data_value_for_index<cfg_int64_t>(di, _frameNumber[m]);}
  /** Return data field frameM_timeStampHi of module m in di as cfg_int64_t. m is not range-checked and the index is uninitialized until _set_indexes_data() finds the field. */
  inline cfg_int64_t timeStampHi(XtcData::DataIter& di, unsigned m) {return data_value_for_index<cfg_int64_t>(di, _timeStampHi[m]);}
  /** Return data field frameM_timeStampLo of module m in di as cfg_int64_t. m is not range-checked and the index is uninitialized until _set_indexes_data() finds the field. */
  inline cfg_int64_t timeStampLo(XtcData::DataIter& di, unsigned m) {return data_value_for_index<cfg_int64_t>(di, _timeStampLo[m]);}
  /** Return data field frameM_specialWord of module m in di as cfg_int64_t. m is not range-checked and the index is uninitialized until _set_indexes_data() finds the field. */
  inline cfg_int64_t specialWord(XtcData::DataIter& di, unsigned m) {return data_value_for_index<cfg_int64_t>(di, _specialWord[m]);}

  /** Return data field "frame_shape" of di as Array<cfg_int64_t>, at the index recorded by _set_indexes_data() (0 if not found). */
  inline Array<cfg_int64_t> frame_shape(XtcData::DataIter& di)        {return data_array_for_index<cfg_int64_t>(di, _frame_shape);}
  /** Return data field frameM_data of module m in dd as Array<pnccd_raw_t>. m is not range-checked and the index is uninitialized until _set_indexes_data() finds the field. */
  inline Array<pnccd_raw_t> data2d(XtcData::DescData& dd, unsigned m) {return data_array_for_index<pnccd_raw_t>(dd, _data[m]);}
  /** Return data2d(descdata(di), m). */
  inline Array<pnccd_raw_t> data2d(XtcData::DataIter& di, unsigned m) {return data2d(descdata(di), m);}
  /** Return data field frameM__data of module m in dd as Array<pnccd_raw_t>. m is not range-checked and the index is uninitialized until _set_indexes_data() finds the field. */
  inline Array<pnccd_raw_t> data1d(XtcData::DescData& dd, unsigned m) {return data_array_for_index<pnccd_raw_t>(dd, __data[m]);}
  /** Return data1d(descdata(di), m). */
  inline Array<pnccd_raw_t> data1d(XtcData::DataIter& di, unsigned m) {return data1d(descdata(di), m);}

private:

  index_t _data_Version = 0;         // 1
  index_t _data_TypeId = 0;          // 11
  index_t _Version = 0;              // 2
  index_t _TypeId = 0;               // 12
  index_t _numLinks = 0;             // 4
  index_t _numChannels = 0;          // 1024
  index_t _camexMagic = 0;           // 1599
  index_t _numRows = 0;              // 1024
  index_t _numSubmoduleRows = 0;     // 512
  index_t _payloadSizePerLink = 0;   // 524304
  index_t _numSubmoduleChannels = 0; // 512
  index_t _numSubmodules = 0;        // 4

  index_t _info_shape = 0;
  index_t _timingFName_shape = 0;
  index_t _frame_shape = 0;

  index_t  _data[MAX_NUMBER_OF_MODULES]; // 2-d panel [512][512];
  index_t __data[MAX_NUMBER_OF_MODULES]; // 1-d panel [512*512];

  index_t _frameNumber[MAX_NUMBER_OF_MODULES];
  index_t _timeStampHi[MAX_NUMBER_OF_MODULES];
  index_t _timeStampLo[MAX_NUMBER_OF_MODULES];
  index_t _specialWord[MAX_NUMBER_OF_MODULES];

  char _cbuf1[MAX_NUMBER_OF_MODULES][32];
  char _cbuf2[MAX_NUMBER_OF_MODULES][32];
  char _cbuf3[MAX_NUMBER_OF_MODULES][32];
  char _cbuf4[MAX_NUMBER_OF_MODULES][32];
  char _cbuf5[MAX_NUMBER_OF_MODULES][32];
  char _cbuf6[MAX_NUMBER_OF_MODULES][32];

  NDArray<pnccd_raw_t>   _raw;
  NDArray<pnccd_calib_t> _calib;
  const NDArray<pnccd_pedestals_t>* _peds;

  void _panel_id(std::ostream& os, const int ind);

  void _make_raw(XtcData::DescData& dd);

  //char* panel_ids[MAX_NUMBER_OF_MODULES];
}; // class

} // namespace detector

#endif // PSALG_AREADETECTORPNCCD_H
//-----------------------------
