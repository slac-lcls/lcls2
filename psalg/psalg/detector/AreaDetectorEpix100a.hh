/**
 * @file
 * @brief Declares detector::AreaDetectorEpix100a, an AreaDetector subclass with stub shape and size overrides.
 */
#ifndef PSALG_AREADETECTOREPIX100A_H
#define PSALG_AREADETECTOREPIX100A_H
//-----------------------------

#include "psalg/detector/AreaDetector.hh"

//using namespace std;
using namespace psalg;

namespace detector {

//-----------------------------
/** AreaDetector subclass for Epix100a with stub overrides of shape(evt) and size(evt); everything else comes from AreaDetector. getAreaDetector(detname) creates it for EPIX100A names. */
class AreaDetectorEpix100a : public AreaDetector {
public:

  //------------------------------

  /** Construct as AreaDetector(detname); no ConfigIter is stored. */
  AreaDetectorEpix100a(const std::string& detname);
  /** Destructor; only logs a debug message. */
  virtual ~AreaDetectorEpix100a();

  /** Log an INFO message "In AreaDetectorEpix100a::" followed by msg. */
  void _class_msg(const std::string& msg=std::string());

  /// shape, size, ndim of data from configuration object
  const shape_t* shape(const event_t&);
  //const size_t   ndim (const event_t&);
  /** Stub: logs an INFO message and returns 123. */
  const size_t   size (const event_t&);

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
}; // class

} // namespace detector

#endif // PSALG_AREADETECTOREPIX100A_H
//-----------------------------
