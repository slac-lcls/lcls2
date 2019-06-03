#ifndef PSALG_GEOMETRYOBJECT_H
#define PSALG_GEOMETRYOBJECT_H

//-------------------

#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <math.h>      // sin, cos

#include "psalg/geometry/SegGeometryStore.hh"

//-------------------

using namespace std;

namespace psalg {

/// @addtogroup psalg

/**
 *  @ingroup psalg
 *
 *  @brief Class supports elementary building block for hierarchial geometry description
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @see GeometryObject, CalibFileFinder, psalg/geometry/test/ex_geometry_access.cpp
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
 *    psalg::GeometryObject* geo = new psalg::GeometryObject(pname, 
 *  							         pindex,
 *  							         oname, 
 *  							         oindex,
 *  							         x0,    
 *  							         y0,    
 *  							         z0,    
 *  							         rot_z, 
 *  							         rot_y, 
 *  							         rot_x, 
 *  							         tilt_z,
 *  							         tilt_y,
 *  							         tilt_x );
 *  @endcode
 *
 *  @li Access methods
 *  @code
 *    // get pixel coordinates
 *    const double* X;
 *    const double* Y;
 *    const double* Z;
 *    unsigned   size;
 *    bool       do_tilt=true;
 *    geo->get_pixel_coords(X, Y, Z, size, do_tilt);
 *
 *    // get pixel areas
 *    const double* A;
 *    unsigned   size;
 *    geo->get_pixel_areas(A, size);
 *
 *    // get pixel mask
 *    const int* mask;
 *    unsigned   size;
 *    unsigned   mbits = 377; // 1-edges; 2-wide central cols; 4-non-bound; 8-non-bound neighbours
 *    geo->get_pixel_mask(mask, size, mbits);
 *
 *    shpGO parobj = geo->get_parent();
 *    std::vector<shpGO> lst = geo->get_list_of_children();
 *
 *    std::string oname  = geo->get_geo_name();
 *    unsigned    oindex = geo->get_geo_index();
 *    std::string pname  = geo->get_parent_name();
 *    unsigned    pindex = geo->get_parent_index();
 *    double      pixsize= geo->get_pixel_scale_size();
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

//typedef ndarray<double,1> NDA;
//typedef ndarray<const double,1> CNDA;

//-------------------

class GeometryObject  {
public:

  typedef psalg::SegGeometry SG;

  //typedef GeometryObject* shpGO;
  typedef boost::shared_ptr<GeometryObject> shpGO;

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
  GeometryObject (  std::string pname  = std::string(),
                    unsigned    pindex = 0,
                    std::string oname  = std::string(),
                    unsigned    oindex = 0,
                    double      x0     = 0,
                    double      y0     = 0,
                    double      z0     = 0,
                    double      rot_z  = 0,
                    double      rot_y  = 0,
                    double      rot_x  = 0,                  
                    double      tilt_z = 0,
                    double      tilt_y = 0,
                    double      tilt_x = 0
                  ) ;

  // Destructor
  virtual ~GeometryObject () ;

  std::string string_geo();
  std::string string_geo_children();

  /// Prints info about self object
  void print_geo();

  /// Prints info about children objects
  void print_geo_children();

  /// Sets shared pointer to the parent object
  void set_parent(shpGO parent) { m_parent = parent; }

  /// Adds shared pointer of the children geometry object to the vector
  void add_child (shpGO child) { v_list_of_children.push_back(child); }

  /// Returns shared pointer to the parent geometry object
  shpGO get_parent() { return m_parent; }

  /// Returns vector of shared pointers to children geometry objects
  std::vector<shpGO> get_list_of_children() { return v_list_of_children; }

  /// Returns self object name
  std::string get_geo_name()     { return m_oname; }

  /// Returns self object index
  unsigned    get_geo_index()    { return m_oindex; }

  /// Returns parent object name
  std::string get_parent_name()  { return m_pname; }

  /// Returns parent object index
  unsigned    get_parent_index() { return m_pindex; }

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
  void get_pixel_coords(const double*& X, const double*& Y, const double*& Z, unsigned& size, 
                        const bool do_tilt=true, const bool do_eval=false);

  /**
   *  @brief Returns pointers to pixel areas array
   *  @param[out] areas - pointer to pixel areas array
   *  @param[out] size - size of the pixel areas array (number of pixels)
   */
  void get_pixel_areas(const double*& areas, unsigned& size);

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
  void get_pixel_mask(const int*& mask, unsigned& size, const unsigned& mbits = 0377);

  /// Returns size of geometry object array - number of pixels
  unsigned get_size_geo_array();

  /// Returns pixel scale size of geometry object
  double get_pixel_scale_size();

  /// Returns string of data for output file
  std::string str_data();

  /// Gets self object geometry parameters
  void get_geo_pars( double& x0,
                     double& y0,
                     double& z0,
                     double& rot_z,
                     double& rot_y,
                     double& rot_x,                  
                     double& tilt_z,
                     double& tilt_y,
                     double& tilt_x 
		     );

  /// Sets self object geometry parameters
  void set_geo_pars( const double& x0 = 0,
                     const double& y0 = 0,
                     const double& z0 = 0,
                     const double& rot_z = 0,
                     const double& rot_y = 0,
                     const double& rot_x = 0,                  
                     const double& tilt_z = 0,
                     const double& tilt_y = 0,
                     const double& tilt_x = 0 
		     );

  /// Adds offset for origin of the self object w.r.t. current position
  void move_geo( const double& dx = 0,
                 const double& dy = 0,
                 const double& dz = 0
		 );

  /// Adds tilts to the self object w.r.t. current orientation
  void tilt_geo( const double& dt_x = 0,
                 const double& dt_y = 0,
                 const double& dt_z = 0 
		 );

  /// Delete arrays with allocated memory, reset pointers to 0
  void deallocate_memory();

protected:

private:

  // Data members
  std::string m_pname;
  unsigned    m_pindex;

  std::string m_oname;
  unsigned    m_oindex;

  double      m_x0;
  double      m_y0;
  double      m_z0;

  double      m_rot_z;
  double      m_rot_y;
  double      m_rot_x;

  double      m_tilt_z;
  double      m_tilt_y;
  double      m_tilt_x;

  bool        m_do_tilt;
  unsigned    m_mbits; // mask control bits

  SG* m_seggeom;

  shpGO m_parent;
  std::vector<shpGO> v_list_of_children;

  //ALGO_TYPE m_algo;
  //PC2X1* m_pix_coords_2x1;

  unsigned m_size;
  double*  p_xarr;
  double*  p_yarr;
  double*  p_zarr;
  double*  p_aarr; // pixel area array
  int*     p_marr; // pixel mask array

  void transform_geo_coord_arrays( const double* X, 
                                   const double* Y,  
                                   const double* Z, 
                                   const unsigned size,
                                   double*  Xt,  
                                   double*  Yt,  
                                   double*  Zt,
                                   const bool do_tilt=true
                                  );

  const static double DEG_TO_RAD = 3.141592653589793238463 / 180; 

  static void rotation(const double* X, const double* Y, const unsigned size,
                       const double C, const double S, 
		       double* Xrot, double* Yrot);

  static void rotation(const double* X, const double* Y, const unsigned size, const double angle_deg, 
                       double* Xrot, double* Yrot);

  /// Returns class name for MsgLogger
  static const std::string name() {return "psalg";}

  // Copy constructor and assignment are disabled by default
  GeometryObject ( const GeometryObject& ) ;
  GeometryObject& operator = ( const GeometryObject& ) ;
};

//-------------------

} // namespace psalg

#endif // PSALG_GEOMETRYOBJECT_H

//-------------------
