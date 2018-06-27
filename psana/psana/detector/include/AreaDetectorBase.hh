#ifndef PSANA_AREADETECTORBASE_H
#define PSANA_AREADETECTORBASE_H
//-----------------------------

#include "psalg/include/Logger.h" // MSG, LOGGER
#include "psalg/include/AllocArray.hh"

namespace Detector {

//#include "xtcdata/xtc/Array.hh"

//enum CALIB_TYPE {PEDESTALS=0, PIXEL_RMS, PIXEL_STATUS, PIXEL_GAIN, PIXEL_OFFSET, PIXEL_MASK, PIXEL_BKGD, COMMON_MODE};
//-----------------------------
class AreaDetectorBase {
public:

  //-----------------------------
  typedef uint32_t shape_t;
  typedef uint32_t size_t;
  typedef float    pixel_rms_t;
  typedef float    pixel_bkgd_t;
  typedef uint16_t pixel_mask_t;
  typedef uint16_t pixel_status_t;
  typedef double   common_mode_t;
  typedef float    pedestals_t;
  typedef float    pixel_gain_t;
  typedef float    pixel_offset_t;
  typedef float    pixel_rms_t;
  typedef uint32_t pixel_idx_t;
  typedef float    pixel_coord_t;
  typedef float    pixel_size_t;
  typedef float    tilt_angle_t;

  typedef float    event_t;
  typedef float    raw_t;
  typedef float    calib_t;
  typedef float    image_t;
  typedef float    geometry_t;
  //------------------------------

  AreaDetectorBase(const std::string& detname) : _detname(detname) {}
  virtual ~AreaDetectorBase() {}

  const std::string detname() {return _detname;};

  /// shape, size, ndim of data from configuration object
  virtual const shape_t* shape(event_t&) = 0;
  virtual const size_t   size (event_t&) = 0;
  virtual const size_t   ndim (event_t&) = 0;

  /// access to calibration constants
  virtual const AllocArray<common_mode_t>&   common_mode      (event_t&) = 0;
  virtual const AllocArray<pedestals_t>&     pedestals        (event_t&) = 0;
  virtual const AllocArray<pixel_rms_t>&     rms              (event_t&) = 0;
  virtual const AllocArray<pixel_status_t>&  status           (event_t&) = 0;
  virtual const AllocArray<pixel_gain_t>&    gain             (event_t&) = 0;
  virtual const AllocArray<pixel_offset_t>&  offset           (event_t&) = 0;
  virtual const AllocArray<pixel_bkgd_t>&    background       (event_t&) = 0;
  virtual const AllocArray<pixel_mask_t>&    mask_calib       (event_t&) = 0;
  virtual const AllocArray<pixel_mask_t>&    mask_from_status (event_t&) = 0;
  virtual const AllocArray<pixel_mask_t>&    mask_edges       (event_t&, const size_t& nnbrs=8) = 0;
  virtual const AllocArray<pixel_mask_t>&    mask_neighbors   (event_t&, const size_t& nrows=1, const size_t& ncols=1) = 0;
  virtual const AllocArray<pixel_mask_t>&    mask             (event_t&, const size_t& mbits=0177777) = 0;
  virtual const AllocArray<pixel_mask_t>&    mask             (event_t&, const bool& calib=true,
							                 const bool& sataus=true,
                                                                         const bool& edges=true,
							                 const bool& neighbors=true) = 0;

  /// access to raw, calibrated data, and image
  virtual const AllocArray<raw_t>&   raw  (event_t&) = 0;
  virtual const AllocArray<calib_t>& calib(event_t&) = 0;
  virtual const AllocArray<image_t>& image(event_t&) = 0;
  virtual const AllocArray<image_t>& image(event_t&, const AllocArray<image_t>& nda) = 0;
  virtual const AllocArray<image_t>& array_from_image(event_t&, const AllocArray<image_t>&) = 0;
  virtual void move_geo(event_t&, const pixel_size_t& dx,  const pixel_size_t& dx,  const pixel_size_t& dx) = 0;
  virtual void tilt_geo(event_t&, const tilt_angle_t& dtx, const tilt_angle_t& dtx, const tilt_angle_t& dtx) = 0;

  /// access to geometry
  virtual const geometry_t* geometry(event_t&) = 0;
  virtual const AllocArray<pixel_idx_t>&   indexes    (event_t&, const size_t& axis=0) = 0;
  virtual const AllocArray<pixel_coord_t>& coords     (event_t&, const size_t& axis=0) = 0;
  virtual const AllocArray<pixel_size_t>&  pixel_size (event_t&, const size_t& axis=0) = 0;
  virtual const AllocArray<pixel_size_t>&  image_xaxis(event_t&) = 0;
  virtual const AllocArray<pixel_size_t>&  image_yaxis(event_t&) = 0;

  private:
    std::string _detname;

}; // class

} // namespace Detector

#endif // PSANA_AREADETECTORBASE_H
//-----------------------------
