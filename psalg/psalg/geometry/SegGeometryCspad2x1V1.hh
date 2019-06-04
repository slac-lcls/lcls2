#ifndef PSALG_SEGGEOMETRYCSPAD2X1V1_H
#define PSALG_SEGGEOMETRYCSPAD2X1V1_H

//-------------------

#include "psalg/geometry/SegGeometry.hh"

namespace geometry {

/// @addtogroup geometry

/**
 *  @ingroup geometry
 *
 *  @brief Class SegGeometryCspad2x1V1 defines the cspad 2x1 V1 sensor pixel coordinates in its local frame.
 *
 *
 *  2x1 sensor coordinate frame:
 * 
 *  @code
 *    (Xmin,Ymax)      ^ Y          (Xmax,Ymax)
 *    (0,0)            |            (0,387)
 *       +----------------------------+
 *       |             |              |
 *       |             |              |
 *       |             |              |
 *     --|-------------+--------------|----> X
 *       |             |              |
 *       |             |              |
 *       |             |              |
 *       +----------------------------+
 *    (184,0)          |           (184,387)
 *    (Xmin,Ymin)                  (Xmax,Ymin)
 *  @endcode
 *
 *  Pixel (r,c)=(0,0) is in the top left corner of the matrix which has coordinates (Xmin,Ymax)
 *  Here we assume that 2x1 has 185 rows and 388 columns.
 *  This assumption differs from the DAQ map, where rows and cols are interchanged:
 *  /reg/g/psdm/sw/external/lusi-xtc/2.12.0a/x86_64-rhel5-gcc41-opt/pdsdata/cspad/ElementIterator.hh,
 *  Detector.hh
 *   
 *  @anchor interface
 *  @par<interface> Interface Description
 * 
 *  @li  Include and typedef
 *  @code
 *  #include "psalg/geometry/SegGeometryCspad2x1V1.hh"
 *  typedef geometry::SegGeometryCspad2x1V1 SG2X1;
 *  @endcode
 *
 *  @li  Instatiation
 *  @code
 *       SG2X1 *seg_geom_2x1 = new SG2X1();  
 *  or
 *       bool use_wide_pix_center = true;
 *       SG2X1 *seg_geom_2x1 = new SG2X1(use_wide_pix_center);  
 *  @endcode
 *
 *  @li  Print info
 *  @code
 *       seg_geom_2x1 -> print_seg_info();
 *  @endcode
 *
 *  @li  Access methods
 *  @code
 *        // scalar values
 *        const size_t         array_size        = seg_geom_2x1 -> size(); // 185*388
 *        const size_t         number_of_rows    = seg_geom_2x1 -> rows(); // 185
 *        const size_t         number_of_cols    = seg_geom_2x1 -> cols(); // 388
 *        const pixel_coord_t  pixel_scale_size  = seg_geom_2x1 -> pixel_scale_size();             // 109.92 
 *        const pixel_coord_t  pixel_coord_min   = seg_geom_2x1 -> pixel_coord_min(SG2X1::AXIS_Z);
 *        const pixel_coord_t  pixel_coord_max   = seg_geom_2x1 -> pixel_coord_max(SG2X1::AXIS_X);
 * 
 *        // pointer to arrays with size equal to array_size
 *        const size_t*        p_array_shape     = seg_geom_2x1 -> shape();                        // {185, 388}
 *        const pixel_area_t*  p_pixel_area      = seg_geom_2x1 -> pixel_area_array(); // array of 1-for regular or 2.5-for long pixels
 *        const pixel_coord_t* p_pixel_size_arr  = seg_geom_2x1 -> pixel_size_array(SG2X1::AXIS_X);
 *        const pixel_coord_t* p_pixel_coord_arr = seg_geom_2x1 -> pixel_coord_array(SG2X1::AXIS_Y);
 *
 *        unsigned mbits=0377; // 1-edges; 2-wide central cols; 4-non-bound; 8-non-bound neighbours
 *        const pixel_mask_t*  p_mask_arr = seg_geom_2x1 -> pixel_mask_array(mbits);
 *  @endcode
 *  
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *
 *  @version \$Id$
 *
 *  @author Mikhail S. Dubrovin
 */ 

class SegGeometryCspad2x1V1 : public geometry::SegGeometry {
public:

  /// Number of pixel rows in 2x1 
  static const size_t  ROWS     = 185;

  /// Number of pixel columnss in 2x1
  static const size_t  COLS     = 388;

  /// Half number of pixel columnss in 2x1
  static const size_t  COLSHALF = 194;

  /// Number of pixels in 2x1
  static const size_t  SIZE     = COLS*ROWS; 

  /// Number of corners
  static const size_t  NCORNERS = 4;

  /// Pixel scale size [um] for indexing  
  static const pixel_coord_t PIX_SCALE_SIZE; // = 109.92;

  /// Pixel size [um] in column direction
  static const pixel_coord_t PIX_SIZE_COLS; //  = 109.92;

  /// Pixel size [um] in row direction
  static const pixel_coord_t PIX_SIZE_ROWS; //  = 109.92;

  /// Wide pixel length [um] 
  static const pixel_coord_t PIX_SIZE_WIDE; //  = 274.80;

  /// Pixel size [um] in depth
  static const pixel_coord_t PIX_SIZE_DEPTH; // = 400.;

  /// Conversion factor between um and pix 
  static const double UM_TO_PIX; //             = 1./109.92;

  //-----------------
  /// Implementation of interface methods

  /// Prints segment info for selected bits
  virtual void print_seg_info(const unsigned& pbits=0);

  /// Returns size of the coordinate arrays
  virtual const size_t size() { return SIZE; }

  /// Returns number of rows in segment
  virtual const size_t rows() { return ROWS; }

  /// Returns number of cols in segment
  virtual const size_t cols() { return COLS; }

  /// Returns shape of the segment {rows, cols}
  virtual const size_t* shape() { return &ARR_SHAPE[0]; }

  /// Returns pixel size in um for indexing
  virtual const pixel_coord_t pixel_scale_size() { return PIX_SCALE_SIZE; }

  /// Returns pointer to the array of pixel areas
  virtual const pixel_area_t* pixel_area_array();

  /**  
   *  @brief Returns pointer to the array of pixel size in um for AXIS
   *  @param[in] axis       Axis from the enumerated list for X, Y, and Z axes
   */
  virtual const pixel_coord_t* pixel_size_array(AXIS axis);

  /// Returns pointer to the array of segment pixel coordinates in um for AXIS
  virtual const pixel_coord_t* pixel_coord_array(AXIS axis);

  /// Returns minimal value in the array of segment pixel coordinates in um for AXIS
  virtual const pixel_coord_t pixel_coord_min(AXIS axis);

  /// Returns maximal value in the array of segment pixel coordinates in um for AXIS
  virtual const pixel_coord_t pixel_coord_max(AXIS axis);

  /**  
   *  @brief Returns pointer to the array of pixel mask: 1/0 = ok/masked
   *  @param[in] mbits - mask control bits;
   *             + 1 - mask edges,
   *             + 2 - mask two central columns, 
   *             + 4 - mask non-bounded pixels,
   *             + 8 - mask nearest neighbours of nonbonded pixels. 
   */  
  virtual const pixel_mask_t* pixel_mask_array(const unsigned& mbits = 0377);


  //-----------------
  // Singleton stuff:

  static geometry::SegGeometry* instance(const bool& use_wide_pix_center=false);

private:

  // Constructor
  /**
   *  @brief Fills-in the map of perfect 2x1 coordinates, defined through the chip geometry.
   *  @param[in] use_wide_pix_center Optional parameter can be used if the wide-pixel row coordinate is prefered to be in the raw center.
   */
  SegGeometryCspad2x1V1 (const bool& use_wide_pix_center=false);

  /// Destructor
  virtual ~SegGeometryCspad2x1V1 ();

  static geometry::SegGeometry* m_pInstance;

  //-----------------

  /// Generator of the pixel coordinate arrays.
  void make_pixel_coord_arrs ();

  /// Generator of the pixel size and area arrays.
  void make_pixel_size_arrs ();

  /// Prints class member data
  void print_member_data ();

  /// Prints 2x1 pixel coordinates
  void print_coord_arrs();

  /// Prints minimal and maximal values of the 2x1 coordinates for X, Y, and Z axes
  void print_min_max_coords();


  /// switch between two options of the wide pixel row center
  bool m_use_wide_pix_center;

  /// done bits
  unsigned m_done_bits;

  /// 1-d pixel coordinates of rows and cols
  pixel_coord_t  m_x_rhs_um [COLSHALF];  
  pixel_coord_t  m_x_arr_um [COLS];  
  pixel_coord_t  m_y_arr_um [ROWS];  

  const static size_t IND_CORNER[NCORNERS];
  const static size_t ARR_SHAPE[2];

  /// 2-d pixel coordinate arrays
  pixel_coord_t  m_x_pix_coord_um [ROWS][COLS];  
  pixel_coord_t  m_y_pix_coord_um [ROWS][COLS];  
  pixel_coord_t  m_z_pix_coord_um [ROWS][COLS];

  /// 2-d pixel size arrays
  pixel_coord_t  m_x_pix_size_um [ROWS][COLS];  
  pixel_coord_t  m_y_pix_size_um [ROWS][COLS];  
  pixel_coord_t  m_z_pix_size_um [ROWS][COLS];

  /// 2-d pixel area arrays
  pixel_area_t  m_pix_area_arr [ROWS][COLS];  

  /// 2-d pixel mask arrays
  pixel_mask_t  m_pix_mask_arr [ROWS][COLS];  

  // Copy constructor and assignment are disabled by default
  SegGeometryCspad2x1V1 ( const SegGeometryCspad2x1V1& ) ;
  SegGeometryCspad2x1V1& operator = ( const SegGeometryCspad2x1V1& ) ;
};

} // namespace geometry

#endif // PSALG_SEGGEOMETRYCSPAD2X1V1_H
