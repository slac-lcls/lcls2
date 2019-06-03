//-------------------

#include "psalg/geometry/SegGeometryMatrixV1.hh"
#include <sstream>     // stringstream
#include <math.h>      // sin, cos
#include <iostream>    // cout

using namespace std;

namespace psalg {

//--------------

// Definition of static parameters (reservation of memory)

//const size_t  SegGeometryMatrixV1::ROWS; //     = 512;
//const size_t  SegGeometryMatrixV1::COLS; //     = 512;
//const size_t  SegGeometryMatrixV1::SIZE; //     = COLS*ROWS; 

//const SegGeometry::pixel_coord_t SegGeometryMatrixV1::PIX_SCALE_SIZE = 75;
//const SegGeometry::pixel_coord_t SegGeometryMatrixV1::PIX_SIZE_COLS  = PIX_SCALE_SIZE;
//const SegGeometry::pixel_coord_t SegGeometryMatrixV1::PIX_SIZE_ROWS  = PIX_SCALE_SIZE;
//const SegGeometry::pixel_coord_t SegGeometryMatrixV1::PIX_SIZE_DEPTH = 400.;
//const double                     SegGeometryMatrixV1::UM_TO_PIX      = 1./PIX_SCALE_SIZE;

//const size_t SegGeometryMatrixV1::NCORNERS; //    =   4;
//const size_t SegGeometryMatrixV1::IND_CORNER[NCORNERS] = {0, COLS-1, (ROWS-1)*COLS, ROWS*COLS-1};
//const size_t SegGeometryMatrixV1::ARR_SHAPE[2] = {ROWS, COLS};

//--------------

// Stripe parameters from string like MTRX:512:512:54:54

bool matrix_pars( const std::string& segname
                , size_t& rows
		, size_t& cols
		, float& pix_size_rows
		, float& pix_size_cols)
{
  //std::cout << "segname: " << segname << '\n';

  if(segname.find("MTRX") == std::string::npos) {
    cout << "psalg::matrix_pars - this is not a MTRX segment, segname: " << segname << '\n';
    return false;
  }

  std::string s(segname);
  for(unsigned i=0; i<s.size(); i++) {if(s[i]==':') s[i]=' ';}
  //std::cout << " string: " << s << '\n';

  std::string pref;
  std::stringstream ss(s);
  ss >> pref >> rows >> cols >> pix_size_rows >> pix_size_cols;
  //std::cout << " r: " << rows << " c:" << cols << " row_size:" << pix_size_rows << " col_size:" << pix_size_cols << '\n';
  return true;
}

//-------------------

SegGeometryMatrixV1::SegGeometryMatrixV1 ( const size_t& rows
					 , const size_t& cols
					 , const float& pix_size_rows
					 , const float& pix_size_cols
					 , const float& pix_size_depth
					 , const float& pix_scale_size
                                         )
  : psalg::SegGeometry()
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

  //const pixel_coord_t PIX_SCALE_SIZE_DEF = 12345;
  //const pixel_coord_t PIX_SIZE_COLS_DEF  = 75;
  //const pixel_coord_t PIX_SIZE_ROWS_DEF  = 75;
  //const pixel_coord_t PIX_SIZE_DEPTH_DEF = 400;

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
// Destructor --
//--------------

SegGeometryMatrixV1::~SegGeometryMatrixV1 ()
{
}


//--------------

void SegGeometryMatrixV1::make_pixel_coord_arrs()
{
  if (m_x_arr_um)       delete [] m_x_arr_um;
  if (m_y_arr_um)       delete [] m_y_arr_um;
  if (m_x_pix_coord_um) delete [] m_x_pix_coord_um;
  if (m_y_pix_coord_um) delete [] m_y_pix_coord_um;
  if (m_z_pix_coord_um) delete [] m_z_pix_coord_um;

  m_x_arr_um       = new pixel_coord_t[ROWS];
  m_y_arr_um       = new pixel_coord_t[COLS];
  m_x_pix_coord_um = new pixel_coord_t[SIZE];
  m_y_pix_coord_um = new pixel_coord_t[SIZE];
  m_z_pix_coord_um = new pixel_coord_t[SIZE];

  // Define x-coordinate of pixels
  for (size_t r=0; r<ROWS; r++) m_x_arr_um[r] = r * PIX_SIZE_ROWS;

  // Define y-coordinate of pixels
  for (size_t c=0; c<COLS; c++) m_y_arr_um[c] = c * PIX_SIZE_COLS;

  size_t i = 0;
  for (size_t r=0; r<ROWS; r++) {
    for (size_t c=0; c<COLS; c++) {
      m_x_pix_coord_um [i]   = m_x_arr_um [r];
      m_y_pix_coord_um [i++] = m_y_arr_um [c];
    }
  }

  std::fill_n(m_z_pix_coord_um, int(SIZE), SegGeometry::pixel_coord_t(0));
  m_done_bits ^= 1; // set bit 1
}

//--------------

void SegGeometryMatrixV1::make_pixel_size_arrs()
{
  if (m_done_bits & 2) return;

  if (m_x_pix_size_um) delete [] m_x_pix_size_um;
  if (m_y_pix_size_um) delete [] m_y_pix_size_um;
  if (m_z_pix_size_um) delete [] m_z_pix_size_um;
  if (m_pix_area_arr ) delete [] m_pix_area_arr;

  m_x_pix_size_um = new pixel_coord_t[SIZE];
  m_y_pix_size_um = new pixel_coord_t[SIZE];
  m_z_pix_size_um = new pixel_coord_t[SIZE];
  m_pix_area_arr  = new pixel_area_t [SIZE];

  std::fill_n(m_x_pix_size_um, int(SIZE), PIX_SIZE_COLS);
  std::fill_n(m_y_pix_size_um, int(SIZE), PIX_SIZE_ROWS);
  std::fill_n(m_z_pix_size_um, int(SIZE), PIX_SIZE_DEPTH);
  std::fill_n(m_pix_area_arr,  int(SIZE), SegGeometry::pixel_area_t(1));

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
  for (unsigned counter=0, c=0; c<COLS; c++) {
    cout << " " << m_x_arr_um[c];
    if (++counter > 19) { counter=0; cout << "\n"; }
  }
  cout << "\n"; 

  cout << "m_y_arr_um:\n"; 
  for (unsigned counter=0, r=0; r<ROWS; r++) { 
    cout << " " << m_y_arr_um[r];
    if (++counter > 19) { counter=0; cout << "\n"; }
  }
  cout << "\n"; 
}

//--------------

void SegGeometryMatrixV1::print_min_max_coords()
{
  cout << "Segment coordinate map limits:"
       << "\n  xmin =  " << pixel_coord_min (AXIS_X)
       << "\n  xmax =  " << pixel_coord_max (AXIS_X)
       << "\n  ymin =  " << pixel_coord_min (AXIS_Y)
       << "\n  ymax =  " << pixel_coord_max (AXIS_Y)
       << "\n  zmin =  " << pixel_coord_min (AXIS_Z)
       << "\n  zmax =  " << pixel_coord_max (AXIS_Z)
       << "\n";
}

//--------------
//--------------
//--------------

void SegGeometryMatrixV1::print_seg_info(const unsigned& pbits)
{
  if (pbits & 1) print_member_data();
  if (pbits & 2) print_coord_arrs();
  if (pbits & 4) print_min_max_coords();
}

//--------------

const SegGeometry::pixel_area_t* SegGeometryMatrixV1::pixel_area_array()
{
  make_pixel_size_arrs();

  return m_pix_area_arr;
}

//--------------

const SegGeometry::pixel_coord_t* SegGeometryMatrixV1::pixel_size_array(AXIS axis)
{
  make_pixel_size_arrs();

  if      (axis == AXIS_X) return m_x_pix_size_um;
  else if (axis == AXIS_Y) return m_y_pix_size_um;
  else if (axis == AXIS_Z) return m_z_pix_size_um;
  else                     return m_y_pix_size_um;
}

//--------------

const SegGeometry::pixel_coord_t* SegGeometryMatrixV1::pixel_coord_array (AXIS axis) 
{ 
  if      (axis == AXIS_X) return m_x_pix_coord_um;
  else if (axis == AXIS_Y) return m_y_pix_coord_um;
  else if (axis == AXIS_Z) return m_z_pix_coord_um;
  else                     return m_x_pix_coord_um;
} 

//--------------

const SegGeometry::pixel_coord_t SegGeometryMatrixV1::pixel_coord_min (AXIS axis) 
{ 
  const SegGeometry::pixel_coord_t* arr = pixel_coord_array (axis);
  SegGeometry::pixel_coord_t corner_coords[NCORNERS];
  for (size_t i=0; i<NCORNERS; ++i) { corner_coords[i] = arr[IND_CORNER[i]]; }
  return psalg::min_of_arr(corner_coords, NCORNERS); 
}

//--------------

const SegGeometry::pixel_coord_t SegGeometryMatrixV1::pixel_coord_max (AXIS axis) 
{ 
  const SegGeometry::pixel_coord_t* arr = pixel_coord_array (axis);
  SegGeometry::pixel_coord_t corner_coords[NCORNERS];
  for (size_t i=0; i<NCORNERS; ++i) { corner_coords[i] = arr[IND_CORNER[i]]; }
  return psalg::max_of_arr(corner_coords, NCORNERS); 
}

//--------------

const SegGeometry::pixel_mask_t* SegGeometryMatrixV1::pixel_mask_array(const unsigned& mbits)
{
  //cout << "SegGeometryMatrixV1::pixel_mask_array(): mbits =" << mbits << '\n';   

  if ( !(m_done_bits & 4)) {
     if (m_pix_mask_arr) delete [] m_pix_mask_arr;
     m_pix_mask_arr = new pixel_mask_t[SIZE];
  }

  std::fill_n(m_pix_mask_arr, int(SIZE), SegGeometry::pixel_mask_t(1));

  if(mbits & 1) {
    // mask edges
    for (size_t r=0; r<ROWS; r++) {
      m_pix_mask_arr[r*COLS] = 0;
      m_pix_mask_arr[r*COLS + COLS - 1] = 0;
    }

    for (size_t c=0; c<COLS; c++) {
      m_pix_mask_arr[c] = 0;
      m_pix_mask_arr[(ROWS-1)*COLS + c] = 0;
    }
  } 

  m_done_bits ^= 4; // set bit 3

  return m_pix_mask_arr;
}

//--------------

} // namespace psalg

//--------------
