#ifndef PSALG_AREADETECTORCSPAD_H
#define PSALG_AREADETECTORCSPAD_H
//-----------------------------

#include <stdint.h>  // uint8_t, uint32_t, etc.
#include "psalg/detector/AreaDetector.hh"

//using namespace std;
using namespace psalg;

namespace detector {

typedef int16_t raw_cspad_t; // raw daq rata type of cspad

//-----------------------------
class AreaDetectorCspad : public AreaDetector {
public:

  //------------------------------
  void process_config();
  void process_data(XtcData::DataIter& datao);
  void detid(std::ostream& os, const int ind=-1); //ind for panel, -1-for entire detector 

  const void print_config();

  AreaDetectorCspad(const std::string& detname, XtcData::ConfigIter& config);
  AreaDetectorCspad(const std::string& detname);
  virtual ~AreaDetectorCspad();

  void _class_msg(const std::string& msg=std::string());

  NDArray<raw_cspad_t>& raw(XtcData::DescData& ddata);
  NDArray<raw_cspad_t>& raw(XtcData::DataIter& datao);

  // implemented in AreaDetector
  /// shape, size, ndim of data from configuration object
  //const size_t   ndim (); // defiled in superclass AreaDetector
  //const size_t   size ();
  //const shape_t* shape();

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

  enum {MAX_NUMBER_OF_MODULES=8};

private:

  int64_cfg_t moduleVersion  [MAX_NUMBER_OF_MODULES];
  int64_cfg_t firmwareVersion[MAX_NUMBER_OF_MODULES];
  int64_cfg_t serialNumber   [MAX_NUMBER_OF_MODULES];

  NDArray<raw_cspad_t> _raw;

  //void _panel_id(std::ostream& os, const int ind);

  //char* panel_ids[MAX_NUMBER_OF_MODULES];
}; // class

} // namespace detector

#endif // PSALG_AREADETECTORCSPAD_H
//-----------------------------
