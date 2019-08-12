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

class CalibPars {

public:

  //typedef geometry::GeometryObject::SG SG;
  typedef geometry::AXIS AXIS;

  CalibPars(const char* detname = "Undefined detname", const DBTYPE& dbtype=DBWEB);
  virtual ~CalibPars();

  const std::string& detname() {return _detname;}

  //-------------------

  /// access to calibration constants
  virtual const NDArray<common_mode_t>&   common_mode      (Query&);
  virtual const NDArray<pedestals_t>&     pedestals        (Query&);
  virtual const NDArray<double>&          pedestals_d      (Query&);
  virtual const NDArray<pixel_rms_t>&     rms              (Query&);
  virtual const NDArray<pixel_status_t>&  status           (Query&);
  virtual const NDArray<pixel_gain_t>&    gain             (Query&);
  virtual const NDArray<pixel_offset_t>&  offset           (Query&);
  virtual const NDArray<pixel_bkgd_t>&    background       (Query&);
  virtual const NDArray<pixel_mask_t>&    mask_calib       (Query&);
  virtual const NDArray<pixel_mask_t>&    mask_from_status (Query&);
  virtual const NDArray<pixel_mask_t>&    mask_edges       (Query&);//, const size_t& nnbrs=8);
  virtual const NDArray<pixel_mask_t>&    mask_neighbors   (Query&);//, const size_t& nrows=1, const size_t& ncols=1);
  virtual const NDArray<pixel_mask_t>&    mask_bits        (Query&);// q.MASKBITS
  virtual const NDArray<pixel_mask_t>&    mask             (Query&);//, const bool& calib=true,
							                  //  const bool& status=true,
                                                                          //  const bool& edges=true,
							                  //  const bool& neighbors=true);

  /// access to geometry
  geometry::GeometryAccess* geometryAccess(Query&);
  void deleteGeometryAccess();

  //virtual const geometry_t& geometry(Query&);
  virtual const geometry_t& geometry_str(Query&); // returns geometry calibration file content as string

  virtual NDArray<const pixel_coord_t>& coords     (Query&); // q.AXISNUM
  virtual NDArray<const pixel_idx_t>&   indexes    (Query&); // q.AXISNUM
  virtual NDArray<const pixel_size_t>&  pixel_size (Query&); // q.AXISNUM
  virtual NDArray<const pixel_area_t>&  pixel_area (Query&);
  virtual NDArray<const pixel_mask_t>&  mask_geo   (Query&); // q.MASKBITSGEO

  virtual NDArray<const pixel_size_t>&  image_xaxis(Query&);
  virtual NDArray<const pixel_size_t>&  image_yaxis(Query&);

 //-------------------

  inline CalibParsDB* calibparsdb() {return _calibparsdb;}

  CalibPars(const CalibPars&) = delete;
  CalibPars& operator = (const CalibPars&) = delete;
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
