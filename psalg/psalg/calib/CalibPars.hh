/**
 * @file
 * @brief Declares calib::CalibPars, which returns calibration constants from a CalibParsDB backend and geometry arrays from a GeometryAccess object.
 */
#ifndef PSALG_CALIBPARS_H
#define PSALG_CALIBPARS_H
//-------------------

#include "psalg/calib/NDArray.hh"
#include "psalg/calib/CalibParsTypes.hh"
#include "psalg/calib/CalibParsDBTypes.hh"
#include "psalg/calib/CalibParsDB.hh"
#include "psalg/calib/Query.hh"

#include "psalg/geometry/GeometryAccess.hh"
//#include "psalg/geometry/GeometryObject.hh"

//using namespace std;
using namespace psalg;

namespace calib {

//-------------------

/**
 * Calibration constants and geometry for one detector. Constants come from a CalibParsDB backend created from dbtype by getCalibParsDB().
 * Geometry arrays come from a GeometryAccess object built on first use from the string returned by geometry_str().
 */
class CalibPars {

public:

  //typedef geometry::GeometryObject::SG SG;
  /** Alias for geometry::AXIS. */
  typedef geometry::AXIS AXIS;

  /** Store detname and create the DB backend with getCalibParsDB(dbtype), which gives NULL for types it does not implement. No GeometryAccess is created yet. */
  CalibPars(const char* detname = "Undefined detname", const DBTYPE& dbtype=DBWEB);
  /** Call deleteGeometryAccess(); the DB backend is not deleted. */
  virtual ~CalibPars();

  /** Return the detector name. */
  const std::string& detname() {return _detname;}

  //-------------------

  //template<typename T>
  //virtual const NDArray<T>& get(Query&);

  /// access to calibration constants
  virtual const NDArray<common_mode_t>&   common_mode      (Query&);
  /** Return the DB backend's get_ndarray_float(q) and log a debug message; q is passed unchanged (its CALIBTYPE is not set here). */
  virtual const NDArray<pedestals_t>&     pedestals        (Query&);
  /** Return the DB backend's get_ndarray_double(q) and log a debug message; q is passed unchanged (its CALIBTYPE is not set here). */
  virtual const NDArray<double>&          pedestals_d      (Query&);
  /** Return the DB backend's get_ndarray_float(q) and log a debug message; q is passed unchanged (its CALIBTYPE is not set here). */
  virtual const NDArray<pixel_rms_t>&     rms              (Query&);
  /** Return the DB backend's get_ndarray_uint16(q) and log a debug message; q is passed unchanged (its CALIBTYPE is not set here). */
  virtual const NDArray<pixel_status_t>&  status           (Query&);
  /** Return the DB backend's get_ndarray_float(q) and log a debug message; q is passed unchanged (its CALIBTYPE is not set here). */
  virtual const NDArray<pixel_gain_t>&    gain             (Query&);
  /** Return the DB backend's get_ndarray_float(q) and log a debug message; q is passed unchanged (its CALIBTYPE is not set here). */
  virtual const NDArray<pixel_offset_t>&  offset           (Query&);
  /** Return the DB backend's get_ndarray_float(q) and log a debug message; q is passed unchanged (its CALIBTYPE is not set here). */
  virtual const NDArray<pixel_bkgd_t>&    background       (Query&);
  /** Return the DB backend's get_ndarray_uint16(q) and log a debug message; q is passed unchanged (its CALIBTYPE is not set here). */
  virtual const NDArray<pixel_mask_t>&    mask_calib       (Query&);
  /** Return the DB backend's get_ndarray_uint16(q) and log a debug message; q is passed unchanged (its CALIBTYPE is not set here). */
  virtual const NDArray<pixel_mask_t>&    mask_from_status (Query&);
  /** Return the DB backend's get_ndarray_uint16(q) and log a debug message; q is passed unchanged (its CALIBTYPE is not set here). */
  virtual const NDArray<pixel_mask_t>&    mask_edges       (Query&);//, const size_t& nnbrs=8);
  /** Return the DB backend's get_ndarray_uint16(q) and log a debug message; q is passed unchanged (its CALIBTYPE is not set here). */
  virtual const NDArray<pixel_mask_t>&    mask_neighbors   (Query&);//, const size_t& nrows=1, const size_t& ncols=1);
  /** Return the DB backend's get_ndarray_uint16(q) and log a debug message; q is passed unchanged (its CALIBTYPE is not set here). */
  virtual const NDArray<pixel_mask_t>&    mask_bits        (Query&);// q.MASKBITS
  /** Return the DB backend's get_ndarray_uint16(q) and log a debug message; q is passed unchanged (its CALIBTYPE is not set here). */
  virtual const NDArray<pixel_mask_t>&    mask             (Query&);//, const bool& calib=true,
							                  //  const bool& status=true,
                                                                          //  const bool& edges=true,
							                  //  const bool& neighbors=true);

  /// access to geometry
  geometry::GeometryAccess* geometryAccess(Query&);
  /** Delete the GeometryAccess object if one exists. The pointer is not reset, so a later geometryAccess() call would return the deleted object. */
  void deleteGeometryAccess();

  //virtual const geometry_t& geometry(Query&);
  /** Return the DB backend's get_string(q) and log a debug message; q is passed unchanged. The trailing comment says it is the content of the geometry calibration file. */
  virtual const geometry_t& geometry_str(Query&); // returns geometry calibration file content as string

  /** Return geometryAccess(q)->get_pixel_coords(axis), with axis taken from q's AXISNUM parameter. */
  virtual NDArray<const pixel_coord_t>& coords     (Query&); // q.AXISNUM
  /** Return geometryAccess(q)->get_pixel_coord_indexes(axis), with axis taken from q's AXISNUM parameter. */
  virtual NDArray<const pixel_idx_t>&   indexes    (Query&); // q.AXISNUM
  /** Default stub: logs a warning that the method (named image_size in the message) should be re-implemented in a derived class, and returns an internal array that is never filled. */
  virtual NDArray<const pixel_size_t>&  pixel_size (Query&); // q.AXISNUM
  /** Return geometryAccess(q)->get_pixel_areas(). */
  virtual NDArray<const pixel_area_t>&  pixel_area (Query&);
  /** Return geometryAccess(q)->get_pixel_mask(mbits), with mbits taken from q's MASKBITSGEO parameter. */
  virtual NDArray<const pixel_mask_t>&  mask_geo   (Query&); // q.MASKBITSGEO

  /** Default stub: logs a warning that it should be re-implemented in a derived class and returns an internal array that is never filled. */
  virtual NDArray<const pixel_size_t>&  image_xaxis(Query&);
  /** Default stub: logs a warning that it should be re-implemented in a derived class and returns an internal array that is never filled. */
  virtual NDArray<const pixel_size_t>&  image_yaxis(Query&);

 //-------------------

  /** Return the DB backend pointer (NULL if getCalibParsDB() did not implement the type). */
  inline CalibParsDB* calibparsdb() {return _calibparsdb;}

  /** Copy construction is disabled. */
  CalibPars(const CalibPars&) = delete;
  /** Copy assignment is disabled. */
  CalibPars& operator = (const CalibPars&) = delete;
  /** Default constructor with an empty body: the DB and GeometryAccess pointers are left uninitialized, so destroying such an object deletes an uninitialized pointer. */
  CalibPars(){}

protected:

  NDArray<common_mode_t>  _common_mode;
  NDArray<pedestals_t>    _pedestals;

  NDArray<pixel_rms_t>    _pixel_rms;
  NDArray<pixel_status_t> _pixel_status;
  NDArray<pixel_gain_t>   _pixel_gain;
  NDArray<pixel_offset_t> _pixel_offset;
  NDArray<pixel_bkgd_t>   _pixel_bkgd;
  NDArray<pixel_mask_t>   _pixel_mask;

  //NDArray<const pixel_coord_t>  _pixel_coord;
  //NDArray<const pixel_idx_t>    _pixel_idx;
  NDArray<const pixel_size_t>   _pixel_size;

  geometry_t              _geometry;

  const std::string       _detname;

  CalibParsDB*              _calibparsdb;
  geometry::GeometryAccess* _geometryaccess;

private :

  void _default_msg(const std::string& msg=std::string()) const;

}; // class

//-------------------

} // namespace calib

#endif // PSALG_CALIBPARS_H
