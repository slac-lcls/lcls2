#ifndef PSALG_AREADETECTORJUNGFRAU_H
#define PSALG_AREADETECTORJUNGFRAU_H
//-----------------------------

#include <stdint.h>  // uint8_t, uint32_t, etc.
#include "psalg/detector/AreaDetector.hh"

//using namespace std;
using namespace psalg;

namespace detector {

typedef uint16_t raw_jungfrau_t; // raw daq rata type of jungfrau

//-----------------------------
class AreaDetectorJungfrau : public AreaDetector {
public:

  //------------------------------

  AreaDetectorJungfrau(const std::string& detname, XtcData::ConfigIter& config);
  AreaDetectorJungfrau(const std::string& detname);
  virtual ~AreaDetectorJungfrau();

  virtual void process_config();
  virtual void process_data(XtcData::DataIter& datao);
  virtual void detid(std::ostream& os, const int ind=-1); //ind for panel, -1-for entire detector 

  virtual const void print_config();

  void _class_msg(const std::string& msg=std::string());

  NDArray<raw_jungfrau_t>& raw(XtcData::DescData& ddata);
  NDArray<raw_jungfrau_t>& raw(XtcData::DataIter& datao);

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

  cfg_int64_t _moduleVersion  [MAX_NUMBER_OF_MODULES];
  cfg_int64_t _firmwareVersion[MAX_NUMBER_OF_MODULES];
  cfg_int64_t _serialNumber   [MAX_NUMBER_OF_MODULES];

  NDArray<raw_jungfrau_t> _raw;

  void _panel_id(std::ostream& os, const int ind);

  //char* panel_ids[MAX_NUMBER_OF_MODULES];
}; // class

} // namespace detector

#endif // PSALG_AREADETECTORJUNGFRAU_H
//-----------------------------
