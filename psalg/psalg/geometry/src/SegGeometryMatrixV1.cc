//-------------------

//#include <math.h>      // sin, cos
#include <sstream>     // stringstream
#include <iostream>    // cout

#include "psalg/utils/Logger.hh" // MSG, LOGGER
#include "psalg/geometry/SegGeometryMatrixV1.hh"

using namespace std;

namespace geometry {

//--------------

// Definition of static parameters (reservation of memory)

const pixel_coord_t SegGeometryMatrixV1::PIX_SCALE_SIZE_DEF = 12345;
const pixel_coord_t SegGeometryMatrixV1::PIX_SIZE_COLS_DEF = 75;
const pixel_coord_t SegGeometryMatrixV1::PIX_SIZE_ROWS_DEF = 75;
const pixel_coord_t SegGeometryMatrixV1::PIX_SIZE_DEPTH_DEF = 400.;

//--------------

// Stripe parameters from string like MTRX:512:512:54:54

bool matrix_pars(const std::string& segname
                ,gsize_t& rows
		,gsize_t& cols
		,pixel_coord_t& pix_size_rows
		,pixel_coord_t& pix_size_cols)
{
  if(segname.find("MTRX") == std::string::npos) {
    cout << "geometry::matrix_pars - this is not a MTRX segment, segname: " << segname << '\n';
    return false;
  }

  std::string s(segname);
  for(unsigned i=0; i<s.size(); i++) {if(s[i]==':') s[i]=' ';}

  std::string pref;
  std::stringstream ss(s);
  ss >> pref >> rows >> cols >> pix_size_rows >> pix_size_cols;
  return true;
}

//----------------

// Singleton stuff:
//SegGeometry* SegGeometryMatrixV1::m_pInstance = NULL; // init static pointer for singleton

SegGeometryMatrixV1::MapInstance _map_segname_instance = {};

//----------------


SegGeometry* SegGeometryMatrixV1::instance(const std::string& segname)
{
  gsize_t rows;
  gsize_t cols;
  pixel_coord_t rpixsize;
  pixel_coord_t cpixsize;

  if(! geometry::matrix_pars(segname, rows, cols, rpixsize, cpixsize)) {
    MSG(ERROR, "Can't demangle geometry segment name: " << segname);  
    return 0; // NULL;
  }

  MSG(DEBUG, "segname: " << segname
            << " rows: " << rows << " cols:" << cols 
            << " rpixsize: " << rpixsize << " cpixsize: " << cpixsize);

  //if map does not have segment - add it
  if(geometry::_map_segname_instance.find(segname) == geometry::_map_segname_instance.end()) {
      geometry::_map_segname_instance[segname] = new SegGeometryMatrixV1(rows, cols, rpixsize, cpixsize);
  }                                                                 // pix_size_depth, pix_scale_size);

  return geometry::_map_segname_instance[segname];

  //if(!m_pInstance) m_pInstance = new geometry::SegGeometryMatrixV1(rows, cols, rpixsize, cpixsize); 
  //return m_pInstance;
}

//----------------
// for single instance
/**
SegGeometry* SegGeometryMatrixV1::instance(const std::string& segname)
{
  gsize_t rows;
  gsize_t cols;
  pixel_coord_t rpixsize;
  pixel_coord_t cpixsize;

  if(! geometry::matrix_pars(segname, rows, cols, rpixsize, cpixsize)) {
    MSG(ERROR, "Can't demangle geometry segment name: " << segname);  
    return 0; // NULL;
  }

  MSG(DEBUG, "segname: " << segname
            << " rows: " << rows << " cols:" << cols 
            << " rpixsize: " << rpixsize << " cpixsize: " << cpixsize);

  if(!m_pInstance) m_pInstance = new geometry::SegGeometryMatrixV1(rows, cols, rpixsize, cpixsize); 
                                       // pix_size_depth, pix_scale_size);
  return m_pInstance;
}
*/

//-------------------
//-------------------
//-------------------
//-------------------

SegGeometryMatrixV1::SegGeometryMatrixV1(const gsize_t& rows
					,const gsize_t& cols
					,const pixel_coord_t& pix_size_rows
					,const pixel_coord_t& pix_size_cols
					,const pixel_coord_t& pix_size_depth
					,const pixel_coord_t& pix_scale_size
                                        )
  : geometry::SegGeometry()
  , ROWS(rows)
  , COLS(cols)
  , m_done_bits(0)
  , m_x_arr_um(0)      
  , m_y_arr_um(0)      
  , m_x_pix_coord_um(0)
  , m_y_pix_coord_um(0)
  , m_z_pix_coord_um(0)
  , m_x_pix_size_um(0)
  , m_y_pix_size_um(0) 
  , m_z_pix_size_um(0)
  , m_pix_area_arr(0)
  , m_pix_mask_arr(0)
{
  //cout << "C-tor of SegGeometryMatrixV1" << endl;

  SIZE = ROWS * COLS;
  PIX_SIZE_ROWS  = pix_size_rows;
  PIX_SIZE_COLS  = pix_size_cols;
  PIX_SIZE_DEPTH = pix_size_depth;
  PIX_SCALE_SIZE = (pix_scale_size==SegGeometryMatrixV1::PIX_SCALE_SIZE_DEF)? 
                   min(PIX_SIZE_ROWS, PIX_SIZE_COLS) : pix_scale_size;
  UM_TO_PIX      = 1./PIX_SCALE_SIZE;

  IND_CORNER[0]  = 0;
  IND_CORNER[1]  = COLS-1;
  IND_CORNER[2]  = (ROWS-1)*COLS;
  IND_CORNER[3]  = ROWS*COLS-1;

  ARR_SHAPE[0]   = ROWS;
  ARR_SHAPE[1]   = COLS;

  make_pixel_coord_arrs();
}

//--------------

SegGeometryMatrixV1::~SegGeometryMatrixV1(){}

//--------------

void SegGeometryMatrixV1::make_pixel_coord_arrs()
{
  if(m_x_arr_um)       delete [] m_x_arr_um;
  if(m_y_arr_um)       delete [] m_y_arr_um;
  if(m_x_pix_coord_um) delete [] m_x_pix_coord_um;
  if(m_y_pix_coord_um) delete [] m_y_pix_coord_um;
  if(m_z_pix_coord_um) delete [] m_z_pix_coord_um;

  m_x_arr_um       = new pixel_coord_t[ROWS];
  m_y_arr_um       = new pixel_coord_t[COLS];
  m_x_pix_coord_um = new pixel_coord_t[SIZE];
  m_y_pix_coord_um = new pixel_coord_t[SIZE];
  m_z_pix_coord_um = new pixel_coord_t[SIZE];

  // Define x-coordinate of pixels
  for(gsize_t r=0; r<ROWS; r++) m_x_arr_um[r] = r * PIX_SIZE_ROWS;

  // Define y-coordinate of pixels
  for(gsize_t c=0; c<COLS; c++) m_y_arr_um[c] = c * PIX_SIZE_COLS;

  gsize_t i = 0;
  for(gsize_t r=0; r<ROWS; r++) {
    for(gsize_t c=0; c<COLS; c++) {
      m_x_pix_coord_um[i]   = m_x_arr_um[r];
      m_y_pix_coord_um[i++] = m_y_arr_um[c];
    }
  }

  std::fill_n(m_z_pix_coord_um, int(SIZE), pixel_coord_t(0));
  m_done_bits ^= 1; // set bit 1
}

//--------------

void SegGeometryMatrixV1::make_pixel_size_arrs()
{
  if(m_done_bits & 2) return;

  if(m_x_pix_size_um) delete [] m_x_pix_size_um;
  if(m_y_pix_size_um) delete [] m_y_pix_size_um;
  if(m_z_pix_size_um) delete [] m_z_pix_size_um;
  if(m_pix_area_arr ) delete [] m_pix_area_arr;

  m_x_pix_size_um = new pixel_coord_t[SIZE];
  m_y_pix_size_um = new pixel_coord_t[SIZE];
  m_z_pix_size_um = new pixel_coord_t[SIZE];
  m_pix_area_arr  = new pixel_area_t [SIZE];

  std::fill_n(m_x_pix_size_um, int(SIZE), PIX_SIZE_COLS);
  std::fill_n(m_y_pix_size_um, int(SIZE), PIX_SIZE_ROWS);
  std::fill_n(m_z_pix_size_um, int(SIZE), PIX_SIZE_DEPTH);
  std::fill_n(m_pix_area_arr,  int(SIZE), pixel_area_t(1));

  m_done_bits ^= 2; // set bit 2
}

//--------------

void SegGeometryMatrixV1::print_member_data()
{
  cout << "SegGeometryMatrixV1::print_member_data():"       
       << "\n ROWS                  " << ROWS       
       << "\n COLS                  " << COLS       
       << "\n PIX_SIZE_COLS         " << PIX_SIZE_COLS 
       << "\n PIX_SIZE_ROWS         " << PIX_SIZE_ROWS 
       << "\n PIX_SIZE_UM           " << PIX_SCALE_SIZE
       << "\n UM_TO_PIX             " << UM_TO_PIX
       << "\n";
}

//--------------

void SegGeometryMatrixV1::print_coord_arrs()
{
  cout << "\nSegGeometryMatrixV1::print_coord_arrs\n";

  cout << "m_x_arr_um:\n"; 
  for(unsigned counter=0, c=0; c<COLS; c++) {
    cout << " " << m_x_arr_um[c];
    if(++counter > 19) {counter=0; cout << "\n";}
  }
  cout << "\n"; 

  cout << "m_y_arr_um:\n"; 
  for(unsigned counter=0, r=0; r<ROWS; r++) { 
    cout << " " << m_y_arr_um[r];
    if(++counter > 19) {counter=0; cout << "\n";}
  }
  cout << "\n"; 
}

//--------------

void SegGeometryMatrixV1::print_min_max_coords()
{
  cout << "Segment coordinate map limits:"
       << "\n  xmin =  " << pixel_coord_min(AXIS_X)
       << "\n  xmax =  " << pixel_coord_max(AXIS_X)
       << "\n  ymin =  " << pixel_coord_min(AXIS_Y)
       << "\n  ymax =  " << pixel_coord_max(AXIS_Y)
       << "\n  zmin =  " << pixel_coord_min(AXIS_Z)
       << "\n  zmax =  " << pixel_coord_max(AXIS_Z)
       << "\n";
}

//--------------
//--------------
//--------------

void SegGeometryMatrixV1::print_seg_info(const unsigned& pbits)
{
  if(pbits & 1) print_member_data();
  if(pbits & 2) print_coord_arrs();
  if(pbits & 4) print_min_max_coords();
}

//--------------

const pixel_area_t* SegGeometryMatrixV1::pixel_area_array()
{
  make_pixel_size_arrs();
  return m_pix_area_arr;
}

//--------------

const pixel_coord_t* SegGeometryMatrixV1::pixel_size_array(AXIS axis)
{
  make_pixel_size_arrs();
  if      (axis == AXIS_X) return m_x_pix_size_um;
  else if (axis == AXIS_Y) return m_y_pix_size_um;
  else if (axis == AXIS_Z) return m_z_pix_size_um;
  else                     return m_y_pix_size_um;
}

//--------------

const pixel_coord_t* SegGeometryMatrixV1::pixel_coord_array (AXIS axis) 
{ 
  if      (axis == AXIS_X) return m_x_pix_coord_um;
  else if (axis == AXIS_Y) return m_y_pix_coord_um;
  else if (axis == AXIS_Z) return m_z_pix_coord_um;
  else                     return m_x_pix_coord_um;
} 

//--------------

const pixel_coord_t SegGeometryMatrixV1::pixel_coord_min (AXIS axis) 
{ 
  const pixel_coord_t* arr = pixel_coord_array (axis);
  pixel_coord_t corner_coords[NCORNERS];
  for(gsize_t i=0; i<NCORNERS; ++i) {corner_coords[i] = arr[IND_CORNER[i]];}
  return geometry::min_of_arr(corner_coords, NCORNERS); 
}

//--------------

const pixel_coord_t SegGeometryMatrixV1::pixel_coord_max (AXIS axis) 
{ 
  const pixel_coord_t* arr = pixel_coord_array (axis);
  pixel_coord_t corner_coords[NCORNERS];
  for(gsize_t i=0; i<NCORNERS; ++i) {corner_coords[i] = arr[IND_CORNER[i]];}
  return geometry::max_of_arr(corner_coords, NCORNERS); 
}

//--------------

const pixel_mask_t* SegGeometryMatrixV1::pixel_mask_array(const unsigned& mbits)
{
  //cout << "SegGeometryMatrixV1::pixel_mask_array(): mbits =" << mbits << '\n';   

  if(!(m_done_bits & 4)) {
     if (m_pix_mask_arr) delete [] m_pix_mask_arr;
     m_pix_mask_arr = new pixel_mask_t[SIZE];
  }

  std::fill_n(m_pix_mask_arr, int(SIZE), pixel_mask_t(1));

  if(mbits & 1) {
    // mask edges
    for(gsize_t r=0; r<ROWS; r++) {
      m_pix_mask_arr[r*COLS] = 0;
      m_pix_mask_arr[r*COLS + COLS - 1] = 0;
    }

    for(gsize_t c=0; c<COLS; c++) {
      m_pix_mask_arr[c] = 0;
      m_pix_mask_arr[(ROWS-1)*COLS + c] = 0;
    }
  } 

  m_done_bits ^= 4; // set bit 3

  return m_pix_mask_arr;
}

//--------------

} // namespace geometry

//--------------
