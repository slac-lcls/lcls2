#ifndef PSALG_GEOMETRYOBJECT_H
#define PSALG_GEOMETRYOBJECT_H

//-------------------

// #include <string> // in SegGeometryStore.hh
#include <vector>
#include <math.h>      // sin, cos

#include "psalg/geometry/GeometryTypes.hh"
#include "psalg/geometry/SegGeometryStore.hh"

//-------------------

using namespace std;

namespace geometry {

/// @addtogroup geometry

/**
 *  @ingroup geometry
 *
 *  @brief Class supports elementary building block for hierarchial geometry description
 *
 *  @note This software was developed for the LCLS project.
 *  If you use all or part of it, please give an appropriate acknowledgment.
 *
 *  @author Mikhail S. Dubrovin
 *
 *  @anchor interface
 *  @par<interface> Interface Description
 * 
 *  @li  Include
 *  @code
 *  #include "psalg/geometry/GeometryObject.hh"
 *  typedef boost::shared_ptr<GeometryObject> shpGO;
 *  @endcode
 *
 *  @li Instatiation
 *  \n
 *  @code
 *    geometry::GeometryObject* geo = new geometry::GeometryObject(pname, 
 *  							     pindex,
 *  							     oname, 
 *  							     oindex,
 *  							     x0,    
 *  							     y0,    
 *  							     z0,    
 *  							     rot_z, 
 *  							     rot_y, 
 *  							     rot_x, 
 *  							     tilt_z,
 *  							     tilt_y,
 *  							     tilt_x );
 *  @endcode
 *
 *  @li Access methods
 *  @code
 *    // get pixel coordinates
 *    const pixel_coord_t* X;
 *    const pixel_coord_t* Y;
 *    const pixel_coord_t* Z;
 *    gsize_t size;
 *    bool do_tilt=true;
 *    geo->get_pixel_coords(X, Y, Z, size, do_tilt);
 *
 *    // get pixel areas
 *    const pixel_area_t* A;
 *    gsize_t size;
 *    geo->get_pixel_areas(A, size);
 *
 *    // get pixel mask
 *    const int* mask;
 *    gsize_t size;
 *    unsigned   mbits = 377; // 1-edges; 2-wide central cols; 4-non-bound; 8-non-bound neighbours
 *    geo->get_pixel_mask(mask, size, mbits);
 *
 *    shpGO parobj = geo->get_parent();
 *    std::vector<shpGO> lst = geo->get_list_of_children();
 *
 *    std::string oname  = geo->get_geo_name();
 *    segindex_t  oindex = geo->get_geo_index();
 *    std::string pname  = geo->get_parent_name();
 *    segindex_t  pindex = geo->get_parent_index();
 *    pixel_coord_t pixsize= geo->get_pixel_scale_size();
 *
 *    double x, y, z, rot_z, rot_y, rot_x, tilt_z, tilt_y, tilt_x;     
 *    geo->get_geo_pars(x, y, z, rot_z, rot_y, rot_x, tilt_z, tilt_y, tilt_x);
 *  
 *    // Next methods are used in class GeometryAccess for building of hierarchial geometry structure.
 *    geo->set_parent(parent_geo);
 *    geo->add_child(child_geo);
 *  @endcode
 *
 *  @li Modification methods
 *  @code
 *    geo->set_geo_pars( 10, 11, 12, 90, 0, 0, 0, 0, 0)
 *    geo->move_geo(10, 11, 12);
 *    geo->tilt_geo(0, 0, 0.15);
 *  @endcode
 *  
 *  @li Print methods
 *  @code
 *    geo->print_geo();
 *    geo->print_geo_children();
 *    cout << "Size of geo: " << geo->get_size_geo_array(); 
 *  @endcode
 */

//-------------------


//-------------------

class GeometryObject {
public:

  typedef geometry::SegGeometry SG;
  typedef GeometryObject* pGO;

  /**
   *  @brief Class constructor accepts path to the calibration "geometry" file and verbosity control bit-word 
   *  
   *  @param[in] pname  - parent name
   *  @param[in] pindex - parent index
   *  @param[in] oname  - this object name
   *  @param[in] oindex - this object index
   *  @param[in] x0     - object origin coordinate x[um] in parent frame
   *  @param[in] y0     - object origin coordinate y[um] in parent frame
   *  @param[in] z0     - object origin coordinate z[um] in parent frame
   *  @param[in] rot_z  - object rotation/design angle [deg] around axis z of the parent frame
   *  @param[in] rot_y  - object rotation/design angle [deg] around axis y of the parent frame
   *  @param[in] rot_x  - object rotation/design angle [deg] around axis x of the parent frame
   *  @param[in] tilt_z - object tilt/deviation angle [deg] around axis z of the parent frame
   *  @param[in] tilt_y - object tilt/deviation angle [deg] around axis y of the parent frame
   *  @param[in] tilt_x - object tilt/deviation angle [deg] around axis x of the parent frame
   */
  GeometryObject(std::string   pname  = std::string(),
                 segindex_t    pindex = 0,
                 std::string   oname  = std::string(),
                 segindex_t    oindex = 0,
                 pixel_coord_t x0     = 0,
                 pixel_coord_t y0     = 0,
                 pixel_coord_t z0     = 0,
                 angle_t       rot_z  = 0,
                 angle_t       rot_y  = 0,
                 angle_t       rot_x  = 0,                  
                 angle_t       tilt_z = 0,
                 angle_t       tilt_y = 0,
                 angle_t       tilt_x = 0
                );

  virtual ~GeometryObject() ;

  std::string string_geo();
  std::string string_geo_children();
  /// Returns string of data for output file
  std::string str_data();

  /// Prints info about self object
  void print_geo();

  /// Prints info about children objects
  void print_geo_children();

  /// Sets shared pointer to the parent object
  void set_parent(pGO parent) {m_parent = parent;}

  /// Adds shared pointer of the children geometry object to the vector
  void add_child(pGO child) {v_list_of_children.push_back(child);}

  /// Returns shared pointer to the parent geometry object
  pGO get_parent() {return m_parent;}

  /// Returns vector of shared pointers to children geometry objects
  std::vector<pGO>& get_list_of_children() {return v_list_of_children;}

  /// Returns self object name
  std::string get_geo_name()     {return m_oname;}

  /// Returns self object index
  segindex_t    get_geo_index()    {return m_oindex;}

  /// Returns parent object name
  std::string get_parent_name()  {return m_pname;}

  /// Returns parent object index
  segindex_t    get_parent_index() {return m_pindex;}

  /**
   *  @brief Re-evaluate pixel coordinates (useful if geo is changed)
   *  @param[in]  do_tilt - on/off tilt angle correction
   *  @param[in]  do_eval - enforce (re-)evaluation of pixel coordinates
   */
  void evaluate_pixel_coords(const bool do_tilt=true, const bool do_eval=false);

  /**
   *  @brief Returns pointers to pixel coordinate arrays
   *  @param[out] X - pointer to x pixel coordinate array
   *  @param[out] Y - pointer to y pixel coordinate array
   *  @param[out] Z - pointer to z pixel coordinate array
   *  @param[out] size - size of the pixel coordinate array (number of pixels)
   *  @param[in]  do_tilt - on/off tilt angle correction
   *  @param[in]  do_eval - enforce (re-)evaluation of pixel coordinates
   */
  void get_pixel_coords(const pixel_coord_t*& X, const pixel_coord_t*& Y, const pixel_coord_t*& Z, gsize_t& size, 
                        const bool do_tilt=true, const bool do_eval=false);

  /**
   *  @brief Returns pointers to pixel areas array
   *  @param[out] areas - pointer to pixel areas array
   *  @param[out] size - size of the pixel areas array (number of pixels)
   */
  void get_pixel_areas(const pixel_area_t*& areas, gsize_t& size);

  /**
   *  @brief Returns pointers to pixel mask array
   *  @param[out] mask - pointer to pixel mask array
   *  @param[out] size - size of the pixel mask array (number of pixels)
   *  @param[in]  mbits - mask control bits; 
   *              + 1 - mask edges,
   *              + 2 - mask two central columns, 
   *              + 4 - mask non-bounded pixels,
   *              + 8 - mask nearest neighbours of nonbonded pixels.
   */
  void get_pixel_mask(const pixel_mask_t*& mask, gsize_t& size, const unsigned& mbits = 0377);

  /// Returns size of geometry object array - number of pixels
  gsize_t get_size_geo_array();

  /// Returns pixel scale size of geometry object
  pixel_coord_t get_pixel_scale_size();

  /// Gets self object geometry parameters
  void get_geo_pars(pixel_coord_t& x0,
                    pixel_coord_t& y0,
                    pixel_coord_t& z0,
                    angle_t&       rot_z,
                    angle_t&       rot_y,
                    angle_t&       rot_x,                  
                    angle_t&       tilt_z,
                    angle_t&       tilt_y,
                    angle_t&       tilt_x 
		   );

  /// Sets self object geometry parameters
  void set_geo_pars(const pixel_coord_t& x0 = 0,
                    const pixel_coord_t& y0 = 0,
                    const pixel_coord_t& z0 = 0,
                    const angle_t&       rot_z = 0,
                    const angle_t&       rot_y = 0,
                    const angle_t&       rot_x = 0,                  
                    const angle_t&       tilt_z = 0,
                    const angle_t&       tilt_y = 0,
                    const angle_t&       tilt_x = 0 
		   );

  /// Adds offset for origin of the self object w.r.t. current position
  void move_geo(const pixel_coord_t& dx = 0,
                const pixel_coord_t& dy = 0,
                const pixel_coord_t& dz = 0
	       );

  /// Adds tilts to the self object w.r.t. current orientation
  void tilt_geo(const pixel_coord_t& dt_x = 0,
                const pixel_coord_t& dt_y = 0,
                const pixel_coord_t& dt_z = 0 
	       );

  /// Returns class name for MsgLogger
  static const std::string name() {return "GeometryObject";}

protected:

private:

  // Data members
  std::string   m_pname;
  segindex_t    m_pindex;

  std::string   m_oname;
  segindex_t    m_oindex;

  pixel_coord_t m_x0;
  pixel_coord_t m_y0;
  pixel_coord_t m_z0;

  angle_t       m_rot_z;
  angle_t       m_rot_y;
  angle_t       m_rot_x;

  angle_t       m_tilt_z;
  angle_t       m_tilt_y;
  angle_t       m_tilt_x;

  bool          m_do_tilt;
  unsigned      m_mbits; // mask control bits

  SG* m_seggeom;

  pGO m_parent;
  std::vector<pGO> v_list_of_children;

  gsize_t m_size;
  pixel_coord_t* p_xarr;
  pixel_coord_t* p_yarr;
  pixel_coord_t* p_zarr;
  pixel_area_t*  p_aarr; // pixel area array
  pixel_mask_t*  p_marr; // pixel mask array

  void transform_geo_coord_arrays(const pixel_coord_t* X,
				  const pixel_coord_t* Y,
				  const pixel_coord_t* Z,
				  const gsize_t size,
				  pixel_coord_t* Xt,
				  pixel_coord_t* Yt,
				  pixel_coord_t* Zt,
				  const bool do_tilt=true
				  );

  static void rotation(const pixel_coord_t* X, const pixel_coord_t* Y, const gsize_t size,
                       const double C, const double S,
		       pixel_coord_t* Xrot, pixel_coord_t* Yrot);

  static void rotation(const pixel_coord_t* X, const pixel_coord_t* Y, const gsize_t size, const angle_t angle_deg,
                       pixel_coord_t* Xrot, pixel_coord_t* Yrot);

  /// Delete arrays with allocated memory, reset pointers to 0
  void _deallocate_memory();

  // Copy constructor and assignment are disabled by default
  GeometryObject(const GeometryObject&) = delete;
  GeometryObject& operator = (const GeometryObject&) = delete;
};

//-------------------

} // namespace geometry

#endif // PSALG_GEOMETRYOBJECT_H

//-------------------
