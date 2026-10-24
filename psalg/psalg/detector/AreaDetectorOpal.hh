/**
 * @file
 * @brief Declares detector::AreaDetectorOpal, the AreaDetector subclass for Opal config and data fields, and its element-type aliases.
 */
#ifndef PSALG_AREADETECTOROPAL_H
#define PSALG_AREADETECTOROPAL_H
//-----------------------------

#include <stdint.h>  // uint8_t, uint32_t, etc.
#include "psalg/detector/AreaDetector.hh"

//using namespace std;
using namespace psalg;

namespace detector {

/** uint16_t; element type of AreaDetectorOpal::raw(). */
typedef uint16_t opal_raw_t;
/** float; element type of AreaDetectorOpal::calib(). */
typedef float    opal_calib_t;
/** double; element type of the pedestals that calib() subtracts (loaded by load_calib_constants()). */
typedef double   opal_pedestals_t;

//-----------------------------
/** AreaDetector subclass for Opal: finds the indexes of the Opal config and data fields by name, gives typed accessors for them, and provides raw() and pedestal-subtracted calib() arrays. */
class AreaDetectorOpal : public AreaDetector {
public:

  //------------------------------

  /** Construct with AreaDetector(detname, configiter) and call _set_indexes_config(configiter). */
  AreaDetectorOpal(const std::string& detname, XtcData::ConfigIter& configiter);
  /** Construct with AreaDetector(detname); no ConfigIter is stored. The trailing comment says _set_indexes_config(ci) is needed afterwards, but that method returns at once when no ConfigIter is stored. */
  AreaDetectorOpal(const std::string& detname); // needs in det._set_indexes_config(ci);
  /** Default constructor (AreaDetector()). */
  AreaDetectorOpal();
  /** Destructor; only logs a debug message. */
  virtual ~AreaDetectorOpal();

  /** Write ind to os zero-padded to a width of 4 characters. */
  virtual void detid(std::ostream& os, const int ind=-1); //ind for panel, -1-for entire detector

  /** Return at once if no ConfigIter is stored. Otherwise store configiter, record the index of each known config field (Version, TypeId, ..., output_lookup_table) by name, and set maxModulesPerDetector = 1, numberOfModules = 1, numberOfRows = Row_Pixels(), numberOfColumns = Column_Pixels() and numberOfPixels to their product. */
  virtual void _set_indexes_config(XtcData::ConfigIter&);
  /**
   * Return at once if a DataIter was already recorded. Otherwise record dataiter, print each data field (index, name, rank, type) to stdout, and record the index of each known data field (data_Version, ..., data16) by name.
   * Uses the NamesLookup of the stored ConfigIter.
   */
  virtual void _set_indexes_data(XtcData::DataIter&);
  //virtual void _set_indexes_data(XtcData::DescData&);

  /** Print the recorded index of every config field to stdout. */
  virtual const void print_config_indexes();
  /** Print detname, dettype, every config value, the derived counts, detid(), ndim() and size() to stdout. */
  virtual const void print_config();
  /** Call _set_indexes_data(di) if no DataIter is recorded yet, then print the recorded data-field indexes to stdout (the first two lines show the config indexes of Version and TypeId). */
  virtual const void print_data_indexes(XtcData::DataIter&);
  /** Call _set_indexes_data(di) if no DataIter is recorded yet, then print every data value and array of di to stdout. */
  virtual const void print_data(XtcData::DataIter&);

  /** Log an INFO message "In AreaDetectorOpal::" followed by msg. */
  void _class_msg(const std::string& msg=std::string());

  /**
   * Reserve size() elements in the internal raw array, set its shape from shape() and ndim(), and return it.
   * No data is copied from dd (the copy code in AreaDetectorOpal.cc is commented out), so the array contents are not filled from the event.
   */
  virtual NDArray<opal_raw_t>& raw(XtcData::DescData&);
  /** Call _set_indexes_data(di), then return raw(descdata(di)). */
  virtual NDArray<opal_raw_t>& raw(XtcData::DataIter& di) {_set_indexes_data(di); return raw(descdata(di));}

  /** Forward to AreaDetector::calib(evt), a default that logs a warning; declared here to avoid a hidden-overload warning. */
  virtual const NDArray<calib_t>& calib(const event_t& evt) {return AreaDetector::calib(evt);} // Silence compiler "hidden" warning
  /**
   * Prepare the raw array as raw(dd) does, then fill and return an internal float array with raw minus pedestals, element by element.
   * The pedestals pointer is set only by load_calib_constants(); it is not checked.
   */
  virtual NDArray<opal_calib_t>& calib(XtcData::DescData&);
  /** Call _set_indexes_data(di), then return calib(descdata(di)). */
  virtual NDArray<opal_calib_t>& calib(XtcData::DataIter& di) {_set_indexes_data(di); return calib(descdata(di));}

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

  // CONFIGURATION values, arrays specific for AreaDetectorOpal
  inline int64_t Version                        () {return config_value_for_index<int64_t>(_Version);}
  /** Return config field "TypeId" as int64_t, at the index recorded by _set_indexes_config() (0 if not found); aborts if that field is not INT64. */
  inline int64_t TypeId                         () {return config_value_for_index<int64_t>(_TypeId);}
  /** Return config field "defect_pixel_correction_enabled" as int64_t, at the index recorded by _set_indexes_config() (0 if not found); aborts if that field is not INT64. */
  inline int64_t defect_pixel_correction_enabled() {return config_value_for_index<int64_t>(_defect_pixel_correction_enabled);}
  /** Return config field "number_of_defect_pixels" as int64_t, at the index recorded by _set_indexes_config() (0 if not found); aborts if that field is not INT64. */
  inline int64_t number_of_defect_pixels        () {return config_value_for_index<int64_t>(_number_of_defect_pixels        );}
  /** Return config field "output_offset" as int64_t, at the index recorded by _set_indexes_config() (0 if not found); aborts if that field is not INT64. */
  inline int64_t output_offset                  () {return config_value_for_index<int64_t>(_output_offset                  );}
  /** Return config field "gain_percent" as int64_t, at the index recorded by _set_indexes_config() (0 if not found); aborts if that field is not INT64. */
  inline int64_t gain_percent                   () {return config_value_for_index<int64_t>(_gain_percent                   );}
  /** Return config field "Column_Pixels" as int64_t, at the index recorded by _set_indexes_config() (0 if not found); aborts if that field is not INT64. */
  inline int64_t Column_Pixels                  () {return config_value_for_index<int64_t>(_Column_Pixels                  );}
  /** Return config field "Row_Pixels" as int64_t, at the index recorded by _set_indexes_config() (0 if not found); aborts if that field is not INT64. */
  inline int64_t Row_Pixels                     () {return config_value_for_index<int64_t>(_Row_Pixels                     );}
  /** Return config field "Mirroring" as int64_t, at the index recorded by _set_indexes_config() (0 if not found); aborts if that field is not INT64. */
  inline int64_t Mirroring                      () {return config_value_for_index<int64_t>(_Mirroring                      );}
  /** Return config field "output_mirroring" as int64_t, at the index recorded by _set_indexes_config() (0 if not found); aborts if that field is not INT64. */
  inline int64_t output_mirroring               () {return config_value_for_index<int64_t>(_output_mirroring               );}
  /** Return config field "vertical_binning" as int64_t, at the index recorded by _set_indexes_config() (0 if not found); aborts if that field is not INT64. */
  inline int64_t vertical_binning               () {return config_value_for_index<int64_t>(_vertical_binning               );}
  /** Return config field "Depth" as int64_t, at the index recorded by _set_indexes_config() (0 if not found); aborts if that field is not INT64. */
  inline int64_t Depth                          () {return config_value_for_index<int64_t>(_Depth                          );}
  /** Return config field "Output_LUT_Size" as int64_t, at the index recorded by _set_indexes_config() (0 if not found); aborts if that field is not INT64. */
  inline int64_t Output_LUT_Size                () {return config_value_for_index<int64_t>(_Output_LUT_Size                );}
  /** Return config field "Binning" as int64_t, at the index recorded by _set_indexes_config() (0 if not found); aborts if that field is not INT64. */
  inline int64_t Binning                        () {return config_value_for_index<int64_t>(_Binning                        );}
  /** Return config field "output_resolution" as int64_t, at the index recorded by _set_indexes_config() (0 if not found); aborts if that field is not INT64. */
  inline int64_t output_resolution              () {return config_value_for_index<int64_t>(_output_resolution              );}
  /** Return config field "output_resolution_bits" as int64_t, at the index recorded by _set_indexes_config() (0 if not found); aborts if that field is not INT64. */
  inline int64_t output_resolution_bits         () {return config_value_for_index<int64_t>(_output_resolution_bits         );}
  /** Return config field "vertical_remapping" as int64_t, at the index recorded by _set_indexes_config() (0 if not found); aborts if that field is not INT64. */
  inline int64_t vertical_remapping             () {return config_value_for_index<int64_t>(_vertical_remapping             );}
  /** Return config field "LUT_Size" as int64_t, at the index recorded by _set_indexes_config() (0 if not found); aborts if that field is not INT64. */
  inline int64_t LUT_Size                       () {return config_value_for_index<int64_t>(_LUT_Size                       );}
  /** Return config field "output_lookup_table_enabled" as int64_t, at the index recorded by _set_indexes_config() (0 if not found); aborts if that field is not INT64. */
  inline int64_t output_lookup_table_enabled    () {return config_value_for_index<int64_t>(_output_lookup_table_enabled    );}
  /** Return config field "black_level" as int64_t, at the index recorded by _set_indexes_config() (0 if not found); aborts if that field is not INT64. */
  inline int64_t black_level                    () {return config_value_for_index<int64_t>(_black_level                    );}

  /** Return config field "output_lookup_table" as Array<uint16_t>, at the index recorded by _set_indexes_config() (0 if not found). */
  inline Array<uint16_t> output_lookup_table    () {return config_array_for_index<uint16_t>(_output_lookup_table);}

  //DATA values, arrays specific for AreaDetectorOpal
  /** Return data field "data_Version" of di as int64_t, at the index recorded by _set_indexes_data() (0 if not found); aborts if that field is not INT64. */
  inline int64_t data_Version(XtcData::DataIter& di) {return data_value_for_index<int64_t>(di, _data_Version);}
  /** Return data field "data_TypeId" of di as int64_t, at the index recorded by _set_indexes_data() (0 if not found); aborts if that field is not INT64. */
  inline int64_t data_TypeId (XtcData::DataIter& di) {return data_value_for_index<int64_t>(di, _data_TypeId );}
  /** Return data field "height" of di as int64_t, at the index recorded by _set_indexes_data() (0 if not found); aborts if that field is not INT64. */
  inline int64_t height	     (XtcData::DataIter& di) {return data_value_for_index<int64_t>(di, _height);}
  /** Return data field "width" of di as int64_t, at the index recorded by _set_indexes_data() (0 if not found); aborts if that field is not INT64. */
  inline int64_t width 	     (XtcData::DataIter& di) {return data_value_for_index<int64_t>(di, _width);}
  /** Return data field "depth" of di as int64_t, at the index recorded by _set_indexes_data() (0 if not found); aborts if that field is not INT64. */
  inline int64_t depth 	     (XtcData::DataIter& di) {return data_value_for_index<int64_t>(di, _depth);}
  /** Forward to AreaDetector::offset(evt); declared here to avoid a hidden-overload warning. */
  virtual const NDArray<pixel_offset_t>&  offset(const event_t& evt) {return AreaDetector::offset(evt);}  // Silence compiler "hidden" warning
  /** Return data field "offset" of di as int64_t, at the index recorded by _set_indexes_data() (0 if not found); aborts if that field is not INT64. */
  inline int64_t offset	     (XtcData::DataIter& di) {return data_value_for_index<int64_t>(di, _offset);}
  /** Return data field "depth_bytes" of di as int64_t, at the index recorded by _set_indexes_data() (0 if not found); aborts if that field is not INT64. */
  inline int64_t depth_bytes (XtcData::DataIter& di) {return data_value_for_index<int64_t>(di, _depth_bytes);}

  /** Return data field "_int_pixel_data" of di as Array<uint8_t>, at the index recorded by _set_indexes_data() (0 if not found). */
  inline Array<uint8_t> _int_pixel_data(XtcData::DataIter& di) {return data_array_for_index<uint8_t> (di, __int_pixel_data);}
  /** Return data field "data8" of di as Array<uint8_t>, at the index recorded by _set_indexes_data() (0 if not found). */
  inline Array<uint8_t>  data8         (XtcData::DataIter& di) {return data_array_for_index<uint8_t> (di, _data8);}
  /** Return data field "data16" of di as Array<uint16_t>, at the index recorded by _set_indexes_data() (0 if not found). */
  inline Array<uint16_t> data16        (XtcData::DataIter& di) {return data_array_for_index<uint16_t>(di, _data16);}


private:

  // CONFIGURATION indices
  index_t _Version                         = 0; // rank: 0 type: 7 INT64
  index_t _TypeId                          = 0; //
  index_t _defect_pixel_correction_enabled = 0; //
  index_t _number_of_defect_pixels         = 0; //
  index_t _output_offset                   = 0; //
  index_t _gain_percent                    = 0; //
  index_t _Column_Pixels                   = 0; //
  index_t _Row_Pixels                      = 0; //
  index_t _Mirroring                       = 0; //
  index_t _output_mirroring                = 0; //
  index_t _vertical_binning                = 0; //
  index_t _Depth                           = 0; //
  index_t _Output_LUT_Size                 = 0; //
  index_t _Binning                         = 0; //
  index_t _output_resolution               = 0; //
  index_t _output_resolution_bits          = 0; //
  index_t _vertical_remapping              = 0; //
  index_t _LUT_Size                        = 0; //
  index_t _output_lookup_table_enabled     = 0; //
  index_t _black_level                     = 0; //
  index_t _output_lookup_table             = 0; // rank: 1 type: 1 UINT16,  Array typeid=t ndim=1 size=0 shape=(0)

  // DATA indices
  index_t _data_Version    = 0; // 1
  index_t _data_TypeId     = 0; // 2
  index_t _height          = 0; // 2472
  index_t _width           = 0; // 3296
  index_t _depth           = 0; // 12
  index_t _offset          = 0; // 0
  index_t _depth_bytes     = 0; // 0
  index_t __int_pixel_data = 0; // Array typeid=h ndim=1 size=16295424 shape=(16295424)
  index_t _data8           = 0; // Array typeid=h ndim=2 size=0 shape=(0, 0)
  index_t _data16          = 0; // Array typeid=t ndim=2 size=8147712 shape=(2472, 3296) data=0, 4, 157, 0, ...

  NDArray<opal_raw_t>              _raw;
  NDArray<opal_calib_t>            _calib;
  const NDArray<opal_pedestals_t>* _peds;

  void _panel_id(std::ostream& os, const int ind);
  void _make_raw(XtcData::DescData& dd);

}; // class

} // namespace detector

#endif // PSALG_AREADETECTOROPAL_H
//-----------------------------
