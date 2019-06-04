#ifndef PSALG_SEGGEOMETRYEPIX10KAV1_H
#define PSALG_SEGGEOMETRYEPIX10KAV1_H

//-------------------

#include "psalg/geometry/SegGeometry.hh"

namespace geometry {


/// @addtogroup geometry

/**
 *  @ingroup geometry
 *
 *  @brief Class SegGeometryEpix10kaV1 defines the Epix100 V1 sensor pixel coordinates in its local frame.
 *
 *
 *  2x1 sensor coordinate frame:
 * 
 *  @code
 *  (Xmin,Ymax)      ^ Y          (Xmax,Ymax)
 *  (0,0)            |            (0,383)
 *     ------------------------------
 *     |             |              |
 *     |             |              |
 *     |             |              |
 *     |             |              |
 *     |             |              |
 *     |             |              |
 *     |             |              |
 *   --|-------------+--------------|----> X
 *     |             |              |
 *     |             |              |
 *     |             |              |
 *     |             |              |
 *     |             |              |
 *     |             |              |
 *     |             |              |
 *     ------------------------------
 *  (351,0)          |           (351,383)
 *  (Xmin,Ymin)                  (Xmax,Ymin)
 *  @endcode
 *
 *  Pixel (r,c)=(0,0) is in the top left corner of the matrix which has coordinates (Xmin,Ymax)
 *  Here we assume that segment has 704 rows and 768 columns.
 *  Epix100 has a pixel size 50x50um, 
 *  Epix10k has a pixel size 100x100um,
 *  This assumption differs from the DAQ map, where rows and cols are interchanged:
 *  /reg/g/psdm/sw/external/lusi-xtc/2.12.0a/x86_64-rhel5-gcc41-opt/pdsdata/cspad/ElementIterator.hh,
 *  Detector.hh
 *   
 *  @anchor interface
 *  @par<interface> Interface Description
 * 
 *  @li  Include and typedef
 *  @code
 *  #include "psalg/geometry/SegGeometryEpix10kaV1.hh"
 *  typedef geometry::SegGeometryEpix10kaV1 SG;
 *  @endcode
 *
 *  @li  Instatiation
 *  @code
 *       SG *seg_geom = new SG();  
 *  or
 *       bool use_wide_pix_center = true;
 *       SG *seg_geom = new SG(use_wide_pix_center);  
 *  @endcode
 *
 *  @li  Print info
 *  @code
 *       unsigned pbits=0377; // 1-member data; 2-coordinate arrays; 4-min/max coordinate values
 *       seg_geom -> print_seg_info(pbits);
 *  @endcode
 *
 *  @li  Access methods
 *  @code
 *        // scalar values
 *        const size_t         array_size        = seg_geom -> size();
 *        const size_t         number_of_rows    = seg_geom -> rows();
 *        const size_t         number_of_cols    = seg_geom -> cols();
 *        const pixel_coord_t  pixel_scale_size  = seg_geom -> pixel_scale_size();
 *        const pixel_coord_t  pixel_coord_min   = seg_geom -> pixel_coord_min(SG::AXIS_Z);
 *        const pixel_coord_t  pixel_coord_max   = seg_geom -> pixel_coord_max(SG::AXIS_X);
 * 
 *        // pointer to arrays with size equal to array_size
 *        const size_t*        p_array_shape     = seg_geom -> shape();
 *        const pixel_area_t*  p_pixel_area      = seg_geom -> pixel_area_array(); // array of 1-for regular or 2.5-for long pixels
 *        const pixel_coord_t* p_pixel_size_arr  = seg_geom -> pixel_size_array(SG::AXIS_X);
 *        const pixel_coord_t* p_pixel_coord_arr = seg_geom -> pixel_coord_array(SG::AXIS_Y);
 *
 *        unsigned mbits=0377; // 1-edges; 2-wide central cols; 4-non-bound; 8-non-bound neighbours
 *        const pixel_mask_t*  p_mask_arr = seg_geom -> pixel_mask_array(mbits);
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

class SegGeometryEpix10kaV1 : public geometry::SegGeometry {
public:

  /// Number of pixel rows in segment 
  static const size_t  ROWS     = 352;

  /// Number of pixel columns in segment
  static const size_t  COLS     = 384;

  /// Half number of pixel rows in segment
  static const size_t  ROWSHALF = 176;

  /// Half number of pixel columns in segment
  static const size_t  COLSHALF = 192;

  /// Number of pixels in segment
  static const size_t  SIZE     = COLS*ROWS; 

  /// Number of corners
  static const size_t  NCORNERS = 4;

  /// Pixel scale size [um] for indexing  
  static const pixel_coord_t PIX_SCALE_SIZE; // = 50.0;

  /// Pixel size [um] in column direction
  static const pixel_coord_t PIX_SIZE_COLS; //  = 50.0;

  /// Pixel size [um] in row direction
  static const pixel_coord_t PIX_SIZE_ROWS; //  = 50.0;

  /// Wide pixel length [um] 
  static const pixel_coord_t PIX_SIZE_WIDE; //  = 200.0;

  /// Pixel size [um] in depth
  static const pixel_coord_t PIX_SIZE_DEPTH; // = 400.;

  /// Conversion factor between um and pix 
  static const double UM_TO_PIX; //             = 1./50;

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
   */  
  virtual const pixel_mask_t* pixel_mask_array(const unsigned& mbits = 0377);

  //-----------------
  // Singleton stuff:

  static geometry::SegGeometry* instance(const bool& use_wide_pix_center=false);

private:

  // Constructor
  /**
   *  @brief Fills-in the map of perfect segment coordinates, defined through the chip geometry.
   *  @param[in] use_wide_pix_center Optional parameter can be used if the wide-pixel row coordinate is prefered to be in the raw center.
   */
  SegGeometryEpix10kaV1 (const bool& use_wide_pix_center=false);

  /// Destructor
  virtual ~SegGeometryEpix10kaV1 ();

  static geometry::SegGeometry* m_pInstance;

  //-----------------

  /// Generator of the pixel coordinate arrays.
  void make_pixel_coord_arrs ();

  /// Generator of the pixel size and area arrays.
  void make_pixel_size_arrs ();

  /// Prints class member data
  void print_member_data ();

  /// Prints segment pixel coordinates
  void print_coord_arrs();

  /// Prints minimal and maximal values of the segment coordinates for X, Y, and Z axes
  void print_min_max_coords();


  /// switch between two options of the wide pixel row center
  bool m_use_wide_pix_center;

  /// done bits
  unsigned m_done_bits;

  /// 1-d pixel coordinates of rows and cols
  pixel_coord_t  m_x_rhs_um [COLSHALF];  
  pixel_coord_t  m_y_rhs_um [ROWSHALF];  
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
  SegGeometryEpix10kaV1 ( const SegGeometryEpix10kaV1& ) ;
  SegGeometryEpix10kaV1& operator = ( const SegGeometryEpix10kaV1& ) ;
};

} // namespace geometry

#endif // PSALG_SEGGEOMETRYEPIX10KAV1_H
