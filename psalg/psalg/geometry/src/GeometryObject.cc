//-------------------

#include "psalg/geometry/GeometryObject.hh"

#include <iostream> // for cout
#include <sstream>  // for stringstream
#include <iomanip>  // for setw, setfill
#include <cmath>    // for sqrt, atan2, etc.
#include <cstring>  // for memcpy

#include <omp.h>

//#include <algorithm>    // std::transform
//#include <functional>   // std::plus, minus

#include "psalg/utils/Logger.hh" // MSG, LOGGER
#include "psalg/geometry/UtilsCSPAD.hh"

using namespace std;

namespace geometry {

GeometryObject::GeometryObject(std::string   pname,
                               segindex_t      pindex,
                               std::string   oname,
                               segindex_t      oindex,
                               pixel_coord_t x0,
                               pixel_coord_t y0,
                               pixel_coord_t z0,
                               angle_t       rot_z,
                               angle_t       rot_y,
                               angle_t       rot_x,                  
                               angle_t       tilt_z,
                               angle_t       tilt_y,
                               angle_t       tilt_x 
			      )
    : m_pname  (pname)
    , m_pindex (pindex)
    , m_oname  (oname)
    , m_oindex (oindex)
    , m_x0     (x0)
    , m_y0     (y0)
    , m_z0     (z0)
    , m_rot_z  (rot_z)
    , m_rot_y  (rot_y)
    , m_rot_x  (rot_x)
    , m_tilt_z (tilt_z)
    , m_tilt_y (tilt_y)
    , m_tilt_x (tilt_x)
    , m_do_tilt(true)
    , m_mbits(0377)
    , m_size(0)
    , p_xarr(0)
    , p_yarr(0)
    , p_zarr(0)
    , p_aarr(0)
    , p_marr(0)
{
  m_seggeom = geometry::SegGeometryStore::Create(m_oname);
  m_parent = pGO();
  v_list_of_children.clear();
}

//-------------------

GeometryObject::~GeometryObject()
{
  if (m_seggeom) delete m_seggeom;

  _deallocate_memory();
}

//-------------------

void GeometryObject::_deallocate_memory()
{
  if (p_xarr) {delete [] p_xarr; p_xarr=0;}
  if (p_yarr) {delete [] p_yarr; p_yarr=0;}
  if (p_zarr) {delete [] p_zarr; p_zarr=0;}
  if (p_aarr) {delete [] p_aarr; p_aarr=0;}
  if (p_marr) {delete [] p_marr; p_marr=0;}
}

//-------------------

std::string GeometryObject::string_geo()
{
  std::stringstream ss;
  ss << "parent:"   << std::setw(12) << std::right << m_pname
     << "  pind:"   << std::setw(2)  << m_pindex
     << "  geo:"    << std::setw(12) << m_oname
     << "  oind:"   << std::setw(2)  << m_oindex << std::right << std::fixed
     << "  x0:"     << std::setw(8)  << std::setprecision(0) << m_x0
     << "  y0:"     << std::setw(8)  << std::setprecision(0) << m_y0
     << "  z0:"     << std::setw(8)  << std::setprecision(0) << m_z0
     << "  rot_z:"  << std::setw(6)  << std::setprecision(1) << m_rot_z
     << "  rot_y:"  << std::setw(6)  << std::setprecision(1) << m_rot_y
     << "  rot_x:"  << std::setw(6)  << std::setprecision(1) << m_rot_x
     << "  tilt_z:" << std::setw(8)  << std::setprecision(5) << m_tilt_z
     << "  tilt_y:" << std::setw(8)  << std::setprecision(5) << m_tilt_y
     << "  tilt_x:" << std::setw(8)  << std::setprecision(5) << m_tilt_x;
  return ss.str();
}

//-------------------

std::string GeometryObject::str_data()
{
  std::stringstream ss;
  ss         << std::setw(11) << std::left << m_pname
     << " "  << std::setw(3)  << std::right << m_pindex
     << " "  << std::setw(11) << std::left << m_oname
     << " "  << std::setw(3)  << std::right << m_oindex << std::fixed
     << "  " << std::setw(8)  << std::setprecision(0) << m_x0
     << " "  << std::setw(8)  << std::setprecision(0) << m_y0
     << " "  << std::setw(8)  << std::setprecision(0) << m_z0
     << "  " << std::setw(6)  << std::setprecision(1) << m_rot_z
     << " "  << std::setw(6)  << std::setprecision(1) << m_rot_y
     << " "  << std::setw(6)  << std::setprecision(1) << m_rot_x
     << "  " << std::setw(8)  << std::setprecision(5) << m_tilt_z
     << " "  << std::setw(8)  << std::setprecision(5) << m_tilt_y
     << " "  << std::setw(8)  << std::setprecision(5) << m_tilt_x;
  return ss.str();
}

//-------------------

void GeometryObject::print_geo()
{
  //std::cout << string_geo() << '\n';
  MSG(INFO, string_geo());
}

//-------------------

std::string GeometryObject::string_geo_children()
{
  std::stringstream ss;
  ss << "parent:" << std::setw(12) << std::left << m_pname
     << " i:"     << std::setw(2)  << m_pindex
     << "  geo:"  << std::setw(12) << m_oname
     << " i:"     << std::setw(2)  << m_oindex
     << "  #children:" << v_list_of_children.size();
  for(std::vector<pGO>::iterator it  = v_list_of_children.begin();
                                 it != v_list_of_children.end(); ++ it) {
    ss << " " << (*it)->get_geo_name() << " " << (*it)->get_geo_index();
  }
  return ss.str();
}

//-------------------

void GeometryObject::print_geo_children()
{
  //std::cout << string_geo_children() << '\n';
  MSG(INFO, string_geo_children());
}

//-------------------

void GeometryObject::transform_geo_coord_arrays(const pixel_coord_t* X,
                                                const pixel_coord_t* Y,
                                                const pixel_coord_t* Z,
                                                const gsize_t size,
                                                pixel_coord_t* Xt,
                                                pixel_coord_t* Yt,
                                                pixel_coord_t* Zt,
                                                const bool do_tilt)
{
  // take rotation ( +tilt ) angles in degrees
  angle_t angle_x = (do_tilt) ? m_rot_x + m_tilt_x : m_rot_x;
  angle_t angle_y = (do_tilt) ? m_rot_y + m_tilt_y : m_rot_y;
  angle_t angle_z = (do_tilt) ? m_rot_z + m_tilt_z : m_rot_z;

  // allocate memory for intermediate transformation
  pixel_coord_t* X1 = new pixel_coord_t[size];
  pixel_coord_t* Y1 = new pixel_coord_t[size];
  pixel_coord_t* Z2 = new pixel_coord_t[size];

  // apply three rotations around Z, Y, and X axes
  rotation(X,  Y,  size, angle_z, X1, Y1);
  rotation(Z,  X1, size, angle_y, Z2, Xt);
  rotation(Y1, Z2, size, angle_x, Yt, Zt);

  // apply translation
  for(gsize_t i=0; i<size; ++i) {
    Xt[i] += m_x0;
    Yt[i] += m_y0;
    Zt[i] += m_z0;
  }

  // release allocated memory
  delete [] X1;
  delete [] Y1;
  delete [] Z2;
}

//-------------------

gsize_t GeometryObject::get_size_geo_array()
{
  if(m_seggeom) return m_seggeom -> size();

  gsize_t size=0;
  for(std::vector<pGO>::iterator it  = v_list_of_children.begin();
                                 it != v_list_of_children.end(); ++it) {
    size += (*it)->get_size_geo_array();
  }
  return size;
}

//-------------------

pixel_coord_t GeometryObject::get_pixel_scale_size()
{
  if(m_seggeom) return m_seggeom -> pixel_scale_size();

  pixel_coord_t pixel_scale_size=1;

  for(std::vector<pGO>::iterator it  = v_list_of_children.begin();
                                 it != v_list_of_children.end(); ++it) {
    pixel_scale_size = (*it)->get_pixel_scale_size();
    break;
  }    
  return pixel_scale_size;
}

//-------------------

void GeometryObject::get_pixel_coords(const pixel_coord_t*& X, const pixel_coord_t*& Y, const pixel_coord_t*& Z, gsize_t& size,
                                      const bool do_tilt, const bool do_eval)
{
  // std::cout << "  ============ do_tilt : " << do_tilt << '\n';
  if(p_xarr==0 || do_tilt != m_do_tilt || do_eval) evaluate_pixel_coords(do_tilt, do_eval);
  X    = p_xarr;
  Y    = p_yarr;
  Z    = p_zarr;
  size = m_size;
}

//-------------------

void GeometryObject::get_pixel_areas(const pixel_area_t*& areas, gsize_t& size)
{
  if(p_aarr==0) evaluate_pixel_coords();
  areas = p_aarr;
  size  = m_size;
}

//-------------------

void GeometryObject::get_pixel_mask(const pixel_mask_t*& mask, gsize_t& size, const bitword_t& mbits)
{
  if(mbits != m_mbits or p_marr==0) {m_mbits = mbits; evaluate_pixel_coords();}
  mask = p_marr;
  size = m_size;
}

//-------------------

void GeometryObject::evaluate_pixel_coords(const bool do_tilt, const bool do_eval)
{
  m_do_tilt = do_tilt; 

  gsize_t size = get_size_geo_array();

  if(p_xarr==0 || size != m_size) {
    // allocate memory for pixel coordinate arrays
    m_size = size;
    
    this->_deallocate_memory();
    
    p_xarr = new pixel_coord_t[m_size];
    p_yarr = new pixel_coord_t[m_size];
    p_zarr = new pixel_coord_t[m_size];
    p_aarr = new pixel_area_t [m_size];
    p_marr = new pixel_mask_t [m_size];
  }

  if(m_seggeom) {

       const pixel_coord_t* x_arr = m_seggeom -> pixel_coord_array(AXIS_X);
       const pixel_coord_t* y_arr = m_seggeom -> pixel_coord_array(AXIS_Y);
       const pixel_coord_t* z_arr = m_seggeom -> pixel_coord_array(AXIS_Z);
       const pixel_area_t*  a_arr = m_seggeom -> pixel_area_array();
       const pixel_mask_t*  m_arr = m_seggeom -> pixel_mask_array(m_mbits);

       transform_geo_coord_arrays(x_arr, y_arr, z_arr, m_size, p_xarr, p_yarr, p_zarr, do_tilt);
       std::memcpy(&p_aarr[0], a_arr, m_size*sizeof(pixel_area_t));
       std::memcpy(&p_marr[0], m_arr, m_size*sizeof(pixel_mask_t));
       return;
  }

  segindex_t ibase=0;
  segindex_t ind=0;
  for(std::vector<pGO>::iterator it  = v_list_of_children.begin(); 
                                 it != v_list_of_children.end(); ++it, ++ind) {

    if((*it)->get_geo_index() != ind) {
      std::stringstream ss;
      ss << "WARNING! Geometry object:" << (*it)->get_geo_name() << ":" << (*it)->get_geo_index()
         << " has non-consequtive index at reconstruction: " << ind << '\n';
      //std::cout << ss.str();
      MSG(WARNING, ss.str());
    }

    const pixel_coord_t* pXch; 
    const pixel_coord_t* pYch;
    const pixel_coord_t* pZch; 
    const pixel_area_t* pAch; 
    const pixel_mask_t* pMch; 
    gsize_t      sizech;

    (*it)->get_pixel_coords(pXch, pYch, pZch, sizech, do_tilt, do_eval);       
    transform_geo_coord_arrays(pXch, pYch, pZch, sizech, &p_xarr[ibase], &p_yarr[ibase], &p_zarr[ibase], do_tilt);

    (*it)->get_pixel_areas(pAch, sizech);
    std::memcpy(&p_aarr[ibase], pAch, sizech*sizeof(pixel_area_t));

    (*it)->get_pixel_mask(pMch, sizech, m_mbits);
    std::memcpy(&p_marr[ibase], pMch, sizech*sizeof(pixel_mask_t));

    ibase += sizech;

    (*it)->_deallocate_memory();
  }

  if(ibase == geometry::SIZE2X2 && m_oname == "CSPAD2X2:V1") {
    // shuffle pixels for cspad2x2, geometry::SIZE2X2 = 2*185*388 = 143560 
    // shuffle pixels only once for "CSPAD2X2:V1" only!

    two2x1ToData2x2<pixel_coord_t>(p_xarr);
    two2x1ToData2x2<pixel_coord_t>(p_yarr);
    two2x1ToData2x2<pixel_coord_t>(p_zarr);
    two2x1ToData2x2<pixel_area_t>(p_aarr);
    two2x1ToData2x2<pixel_mask_t>(p_marr);
  }
}

//-------------------

void GeometryObject::get_geo_pars(pixel_coord_t& x0,
                                  pixel_coord_t& y0,
                                  pixel_coord_t& z0,
                                  angle_t& rot_z,
                                  angle_t& rot_y,
                                  angle_t& rot_x,                  
                                  angle_t& tilt_z,
                                  angle_t& tilt_y,
                                  angle_t& tilt_x 
				 )
{
  x0     = m_x0;     
  y0     = m_y0;     
  z0     = m_z0;    
  rot_z  = m_rot_z;  
  rot_y  = m_rot_y; 
  rot_x  = m_rot_x;  
  tilt_z = m_tilt_z;
  tilt_y = m_tilt_y;
  tilt_x = m_tilt_x; 
}

//-------------------

void GeometryObject::set_geo_pars(const pixel_coord_t& x0,
                                  const pixel_coord_t& y0,
                                  const pixel_coord_t& z0,
                                  const angle_t& rot_z,
                                  const angle_t& rot_y,
                                  const angle_t& rot_x,                  
                                  const angle_t& tilt_z,
                                  const angle_t& tilt_y,
                                  const angle_t& tilt_x 
				 )
{
  m_x0     = x0;    
  m_y0     = y0;    
  m_z0     = z0;  
  m_rot_z  = rot_z; 
  m_rot_y  = rot_y;
  m_rot_x  = rot_x; 
  m_tilt_z = tilt_z;
  m_tilt_y = tilt_y;
  m_tilt_x = tilt_x; 
}

//-------------------

void GeometryObject::move_geo(const pixel_coord_t& dx,
                              const pixel_coord_t& dy,
                              const pixel_coord_t& dz
			     )
{
  m_x0 += dx;    
  m_y0 += dy;    
  m_z0 += dz;  
}

//-------------------

void GeometryObject::tilt_geo(const pixel_coord_t& dt_x,
                              const pixel_coord_t& dt_y,
                              const pixel_coord_t& dt_z 
			     )
{
  m_tilt_z += dt_z;
  m_tilt_y += dt_y;
  m_tilt_x += dt_x; 
}

//-------------------
//-------------------
//-------------------
//-------------------

void GeometryObject::rotation(const pixel_coord_t* X, const pixel_coord_t* Y, const gsize_t size,
                              const double C, const double S, pixel_coord_t* Xrot, pixel_coord_t* Yrot)
  {
    // takes 0.48 sec for cspad size=2296960
    /*
    const pixel_coord_t* Xend = &X[size];
    const pixel_coord_t* x=X; // &X[0];
    const pixel_coord_t* y=Y;
    for(; x!=Xend; x++, y++) {
      *Xrot++ = *x*C - *y*S;
      *Yrot++ = *y*C + *x*S;
    }
    */

    // takes 0.55 sec for cspad size=2296960
    // and   0.29 sec with pragma

    omp_set_num_threads(6); // or use env# OMP_NUM_THREADS //num_threads(6); ??

    #pragma omp parallel for
    for(gsize_t i=0; i<size; ++i) {
      Xrot[i] = X[i]*C - Y[i]*S; 
      Yrot[i] = Y[i]*C + X[i]*S; 
    }
  }

//-------------------

/*
void GeometryObject::rotation(const pixel_coord_t* X, const pixel_coord_t* Y, const gsize_t size,
               const double C, const double S, 
               pixel_coord_t* Xrot, pixel_coord_t* Yrot)
  {
    //std::fill_n(Carr, int(size), double(C));
    //std::fill_n(Sarr, int(size), double(S));

    // takes 2 sec

    pixel_coord_t* xc = new pixel_coord_t [size];
    pixel_coord_t* xs = new pixel_coord_t [size];
    pixel_coord_t* yc = new pixel_coord_t [size];
    pixel_coord_t* ys = new pixel_coord_t [size];

    auto vS = std::bind1st(std::multiplies<double>(), S);
    auto vC = std::bind1st(std::multiplies<double>(), C);

    std::transform(X, &X[size], &xs[0], vS);
    std::transform(X, &X[size], &xc[0], vC);
    std::transform(Y, &Y[size], &ys[0], vS);
    std::transform(Y, &Y[size], &yc[0], vC);

    std::transform(&xc[0], &xc[size], &ys[0], Xrot, std::minus<pixel_coord_t>());
    std::transform(&yc[0], &yc[size], &xs[0], Yrot, std::plus<pixel_coord_t>());

    delete [] xc; 
    delete [] xs; 
    delete [] yc; 
    delete [] ys; 
  }
*/

//-------------------

void GeometryObject::rotation(const pixel_coord_t* X, const pixel_coord_t* Y, const gsize_t size,
                              const angle_t angle_deg, pixel_coord_t* Xrot, pixel_coord_t* Yrot)
  {
    const angle_t angle_rad = angle_deg * DEG_TO_RAD;
    const double C = cos(angle_rad);
    const double S = sin(angle_rad);
    rotation(X, Y, size, C, S, Xrot, Yrot);
  }

//-------------------

} // namespace geometry

//-------------------
