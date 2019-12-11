#ifndef PSALG_AREADETECTOROPAL_H
#define PSALG_AREADETECTOROPAL_H
//-----------------------------

#include <stdint.h>  // uint8_t, uint32_t, etc.
#include "psalg/detector/AreaDetector.hh"

//using namespace std;
using namespace psalg;

namespace detector {

typedef uint16_t opal_raw_t;
typedef float    opal_calib_t;
typedef double   opal_pedestals_t;

//-----------------------------
class AreaDetectorOpal : public AreaDetector {
public:

  //------------------------------

  AreaDetectorOpal(const std::string& detname, XtcData::ConfigIter& configiter);
  AreaDetectorOpal(const std::string& detname); // needs in det._set_indexes_config(ci);
  AreaDetectorOpal();
  virtual ~AreaDetectorOpal();

  virtual void detid(std::ostream& os, const int ind=-1); //ind for panel, -1-for entire detector 

  virtual void _set_indexes_config(XtcData::ConfigIter&);
  virtual void _set_indexes_data(XtcData::DataIter&);
  //virtual void _set_indexes_data(XtcData::DescData&);

  virtual const void print_config_indexes();
  virtual const void print_config();
  virtual const void print_data_indexes(XtcData::DataIter&);
  virtual const void print_data(XtcData::DataIter&);

  void _class_msg(const std::string& msg=std::string());

  virtual NDArray<opal_raw_t>& raw(XtcData::DescData&);
  virtual NDArray<opal_raw_t>& raw(XtcData::DataIter& di) {_set_indexes_data(di); return raw(descdata(di));}

  virtual NDArray<opal_calib_t>& calib(XtcData::DescData&);
  virtual NDArray<opal_calib_t>& calib(XtcData::DataIter& di) {_set_indexes_data(di); return calib(descdata(di));}

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
  inline int64_t TypeId                         () {return config_value_for_index<int64_t>(_TypeId);}
  inline int64_t defect_pixel_correction_enabled() {return config_value_for_index<int64_t>(_defect_pixel_correction_enabled);}
  inline int64_t number_of_defect_pixels        () {return config_value_for_index<int64_t>(_number_of_defect_pixels        );}
  inline int64_t output_offset                  () {return config_value_for_index<int64_t>(_output_offset                  );}
  inline int64_t gain_percent                   () {return config_value_for_index<int64_t>(_gain_percent                   );}
  inline int64_t Column_Pixels                  () {return config_value_for_index<int64_t>(_Column_Pixels                  );}
  inline int64_t Row_Pixels                     () {return config_value_for_index<int64_t>(_Row_Pixels                     );}
  inline int64_t Mirroring                      () {return config_value_for_index<int64_t>(_Mirroring                      );}
  inline int64_t output_mirroring               () {return config_value_for_index<int64_t>(_output_mirroring               );}
  inline int64_t vertical_binning               () {return config_value_for_index<int64_t>(_vertical_binning               );}
  inline int64_t Depth                          () {return config_value_for_index<int64_t>(_Depth                          );}
  inline int64_t Output_LUT_Size                () {return config_value_for_index<int64_t>(_Output_LUT_Size                );}
  inline int64_t Binning                        () {return config_value_for_index<int64_t>(_Binning                        );}
  inline int64_t output_resolution              () {return config_value_for_index<int64_t>(_output_resolution              );}
  inline int64_t output_resolution_bits         () {return config_value_for_index<int64_t>(_output_resolution_bits         );}
  inline int64_t vertical_remapping             () {return config_value_for_index<int64_t>(_vertical_remapping             );}
  inline int64_t LUT_Size                       () {return config_value_for_index<int64_t>(_LUT_Size                       );}
  inline int64_t output_lookup_table_enabled    () {return config_value_for_index<int64_t>(_output_lookup_table_enabled    );}
  inline int64_t black_level                    () {return config_value_for_index<int64_t>(_black_level                    );}

  inline Array<uint16_t> output_lookup_table    () {return config_array_for_index<uint16_t>(_output_lookup_table);}

  //DATA values, arrays specific for AreaDetectorOpal
  inline int64_t data_Version(XtcData::DataIter& di) {return data_value_for_index<int64_t>(di, _data_Version);}
  inline int64_t data_TypeId (XtcData::DataIter& di) {return data_value_for_index<int64_t>(di, _data_TypeId );}
  inline int64_t height	     (XtcData::DataIter& di) {return data_value_for_index<int64_t>(di, _height);}
  inline int64_t width 	     (XtcData::DataIter& di) {return data_value_for_index<int64_t>(di, _width);}
  inline int64_t depth 	     (XtcData::DataIter& di) {return data_value_for_index<int64_t>(di, _depth);}
  inline int64_t offset	     (XtcData::DataIter& di) {return data_value_for_index<int64_t>(di, _offset);}
  inline int64_t depth_bytes (XtcData::DataIter& di) {return data_value_for_index<int64_t>(di, _depth_bytes);}

  inline Array<uint8_t> _int_pixel_data(XtcData::DataIter& di) {return data_array_for_index<uint8_t> (di, __int_pixel_data);}
  inline Array<uint8_t>  data8         (XtcData::DataIter& di) {return data_array_for_index<uint8_t> (di, _data8);}
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
