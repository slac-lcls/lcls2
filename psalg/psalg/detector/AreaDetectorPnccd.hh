#ifndef PSALG_AREADETECTORPNCCD_H
#define PSALG_AREADETECTORPNCCD_H
//-----------------------------

#include <stdint.h>  // uint8_t, uint32_t, etc.
#include "psalg/detector/AreaDetector.hh"

//using namespace std;
using namespace psalg;

namespace detector {

typedef uint16_t raw_pnccd_t; // raw   type of pnccd data
typedef  float calib_pnccd_t; // calib type of pnccd data

//-----------------------------
class AreaDetectorPnccd : public AreaDetector {
public:

  //------------------------------

  AreaDetectorPnccd(const std::string& detname, XtcData::ConfigIter& configiter);
  AreaDetectorPnccd(const std::string& detname);
  AreaDetectorPnccd();
  virtual ~AreaDetectorPnccd();

  virtual void detid(std::ostream& os, const int ind=-1); //ind for panel, -1-for entire detector 

  virtual void set_indexes_config(XtcData::ConfigIter&);
  virtual void set_indexes_data(XtcData::DataIter&);

  virtual const void print_config_indexes();
  virtual const void print_data_indexes();
  virtual const void print_config();
  virtual const void print_data(XtcData::DataIter&);

  void _class_msg(const std::string& msg=std::string());

  virtual NDArray<raw_pnccd_t>& raw(XtcData::DescData&);
  virtual NDArray<raw_pnccd_t>& raw(XtcData::DataIter& di) {return raw(descdata(di));}

  virtual NDArray<calib_pnccd_t>& calib(XtcData::DescData&);
  virtual NDArray<calib_pnccd_t>& calib(XtcData::DataIter& di) {return calib(descdata(di));}

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

  enum {MAX_NUMBER_OF_MODULES=4};

  // convenience methods of AreaDetectorPnccd ONLY!
  inline cfg_int64_t Version             () {return config_value_for_index<cfg_int64_t>(*_pconfig, _Version);}
  inline cfg_int64_t TypeId              () {return config_value_for_index<cfg_int64_t>(*_pconfig, _TypeId);}
  inline cfg_int64_t numLinks            () {return config_value_for_index<cfg_int64_t>(*_pconfig, _numLinks);}
  inline cfg_int64_t numChannels         () {return config_value_for_index<cfg_int64_t>(*_pconfig, _numChannels);}
  inline cfg_int64_t camexMagic          () {return config_value_for_index<cfg_int64_t>(*_pconfig, _camexMagic);}
  inline cfg_int64_t numRows             () {return config_value_for_index<cfg_int64_t>(*_pconfig, _numRows);}
  inline cfg_int64_t numSubmoduleRows    () {return config_value_for_index<cfg_int64_t>(*_pconfig, _numSubmoduleRows);}
  inline cfg_int64_t payloadSizePerLink  () {return config_value_for_index<cfg_int64_t>(*_pconfig, _payloadSizePerLink);}
  inline cfg_int64_t numSubmoduleChannels() {return config_value_for_index<cfg_int64_t>(*_pconfig, _numSubmoduleChannels);}
  inline cfg_int64_t numSubmodules       () {return config_value_for_index<cfg_int64_t>(*_pconfig, _numSubmodules);}

  inline Array<cfg_int64_t> info_shape()        {return config_array_for_index<cfg_int64_t>(*_pconfig, _info_shape);}
  inline Array<cfg_int64_t> timingFName_shape() {return config_array_for_index<cfg_int64_t>(*_pconfig, _timingFName_shape);}

  //data values, arrays
  inline cfg_int64_t data_Version (XtcData::DataIter& di) {return data_value_for_index<cfg_int64_t>(di, _data_Version);}
  inline cfg_int64_t data_TypeId  (XtcData::DataIter& di) {return data_value_for_index<cfg_int64_t>(di, _data_TypeId);}
  inline cfg_int64_t data_numLinks(XtcData::DataIter& di) {return data_value_for_index<cfg_int64_t>(di, _numLinks);}
  inline cfg_int64_t frameNumber(XtcData::DataIter& di, unsigned m) {return data_value_for_index<cfg_int64_t>(di, _frameNumber[m]);}
  inline cfg_int64_t timeStampHi(XtcData::DataIter& di, unsigned m) {return data_value_for_index<cfg_int64_t>(di, _timeStampHi[m]);}
  inline cfg_int64_t timeStampLo(XtcData::DataIter& di, unsigned m) {return data_value_for_index<cfg_int64_t>(di, _timeStampLo[m]);}
  inline cfg_int64_t specialWord(XtcData::DataIter& di, unsigned m) {return data_value_for_index<cfg_int64_t>(di, _specialWord[m]);}

  inline Array<cfg_int64_t> frame_shape(XtcData::DataIter& di)        {return data_array_for_index<cfg_int64_t>(di, _frame_shape);}
  inline Array<raw_pnccd_t> data2d(XtcData::DescData& dd, unsigned m) {return data_array_for_index<raw_pnccd_t>(dd, _data[m]);}
  inline Array<raw_pnccd_t> data2d(XtcData::DataIter& di, unsigned m) {return data2d(descdata(di), m);}
  inline Array<raw_pnccd_t> data1d(XtcData::DescData& dd, unsigned m) {return data_array_for_index<raw_pnccd_t>(dd, __data[m]);}
  inline Array<raw_pnccd_t> data1d(XtcData::DataIter& di, unsigned m) {return data1d(descdata(di), m);}

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

  NDArray<raw_pnccd_t>   _raw;
  NDArray<calib_pnccd_t> _calib;
  //const NDArray<double>& _peds;
  NDArray<double> _peds;

  void _panel_id(std::ostream& os, const int ind);

  void _make_raw(XtcData::DescData& dd);

  //char* panel_ids[MAX_NUMBER_OF_MODULES];
}; // class

} // namespace detector

#endif // PSALG_AREADETECTORPNCCD_H
//-----------------------------
