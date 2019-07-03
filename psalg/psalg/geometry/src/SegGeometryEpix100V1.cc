//-----------------------

#include "psalg/geometry/SegGeometryEpix100V1.hh"

//#include <math.h>      // sin, cos
#include <iostream>    // cout

using namespace std;

namespace geometry {

//--------------

// Definition of static parameters (reservation of memory)

const gsize_t SegGeometryEpix100V1::ROWS; //     = 704;
const gsize_t SegGeometryEpix100V1::COLS; //     = 768;
const gsize_t SegGeometryEpix100V1::ROWSHALF; // = 352;
const gsize_t SegGeometryEpix100V1::COLSHALF; // = 384;
const gsize_t SegGeometryEpix100V1::SIZE; //     = COLS*ROWS; 
const gsize_t SegGeometryEpix100V1::NCORNERS; //    =   4;

const pixel_coord_t SegGeometryEpix100V1::PIX_SCALE_SIZE = 50.00;
const pixel_coord_t SegGeometryEpix100V1::PIX_SIZE_COLS  = PIX_SCALE_SIZE;
const pixel_coord_t SegGeometryEpix100V1::PIX_SIZE_ROWS  = PIX_SCALE_SIZE;
const pixel_coord_t SegGeometryEpix100V1::PIX_SIZE_WIDE  = 175.00;
const pixel_coord_t SegGeometryEpix100V1::PIX_SIZE_DEPTH = 400.;
const double        SegGeometryEpix100V1::UM_TO_PIX      = 1./PIX_SCALE_SIZE;

const gsize_t SegGeometryEpix100V1::IND_CORNER[NCORNERS] = {0, COLS-1, (ROWS-1)*COLS, ROWS*COLS-1};
const gsize_t SegGeometryEpix100V1::ARR_SHAPE[2] = {ROWS, COLS};

//----------------
// Singleton stuff:

//SegGeometryEpix100V1*
SegGeometry* SegGeometryEpix100V1::m_pInstance = NULL; // init static pointer for singleton

//SegGeometryEpix100V1*
SegGeometry* SegGeometryEpix100V1::instance(const bool& use_wide_pix_center)
{
  if( !m_pInstance ) m_pInstance = new SegGeometryEpix100V1(use_wide_pix_center);
  return m_pInstance;
}

//----------------

SegGeometryEpix100V1::SegGeometryEpix100V1(const bool& use_wide_pix_center)
  : geometry::SegGeometry()
  , m_use_wide_pix_center(use_wide_pix_center)
  , m_done_bits(0)
{
  //cout << "C-tor of SegGeometryEpix100V1" << endl;

  make_pixel_coord_arrs();
}

//--------------

SegGeometryEpix100V1::~SegGeometryEpix100V1()
{
}

//--------------

void SegGeometryEpix100V1::make_pixel_coord_arrs()
{
  // Define x-coordinate of pixels
  pixel_coord_t x_offset = PIX_SIZE_WIDE - PIX_SIZE_COLS / 2;
  for(gsize_t c=0; c<COLSHALF; c++) m_x_rhs_um[c] = c * PIX_SIZE_COLS + x_offset;
  if(m_use_wide_pix_center)         m_x_rhs_um[0] = PIX_SIZE_WIDE / 2;
  for(gsize_t c=0; c<COLSHALF; c++){m_x_arr_um[c] = -m_x_rhs_um[COLSHALF-1-c];
                                     m_x_arr_um[COLSHALF + c] =  m_x_rhs_um[c];}

  // Define y-coordinate of pixels
  pixel_coord_t y_offset = PIX_SIZE_WIDE - PIX_SIZE_ROWS / 2;
  for(gsize_t r=0; r<ROWSHALF; r++) m_y_rhs_um[r] = r * PIX_SIZE_ROWS + y_offset;
  if(m_use_wide_pix_center)         m_y_rhs_um[0] = PIX_SIZE_WIDE / 2;
  for(gsize_t r=0; r<ROWSHALF; r++){m_y_arr_um[r] = -m_y_rhs_um[ROWSHALF-1-r];
                                     m_y_arr_um[ROWSHALF + r] =  m_y_rhs_um[r];}

  for(gsize_t r=0; r<ROWS; r++) {
    for(gsize_t c=0; c<COLS; c++) {
      m_x_pix_coord_um [r][c] = m_x_arr_um [c];
      m_y_pix_coord_um [r][c] = m_y_arr_um [r];
    }
  }

  std::fill_n(&m_z_pix_coord_um[0][0], int(SIZE), pixel_coord_t(0));
  m_done_bits ^= 1; // set bit 1
}

//--------------

void SegGeometryEpix100V1::make_pixel_size_arrs()
{
  if(m_done_bits & 2) return;

  std::fill_n(&m_x_pix_size_um[0][0], int(SIZE), PIX_SIZE_COLS);
  std::fill_n(&m_y_pix_size_um[0][0], int(SIZE), PIX_SIZE_ROWS);
  std::fill_n(&m_z_pix_size_um[0][0], int(SIZE), PIX_SIZE_DEPTH);
  std::fill_n(&m_pix_area_arr [0][0], int(SIZE), pixel_area_t(1));

  pixel_area_t arr_wide = PIX_SIZE_WIDE/PIX_SIZE_COLS;

  // set size and area for two middle cols of segment
  for(gsize_t r=0; r<ROWS; r++) {
    m_x_pix_size_um[r][COLSHALF-1] = PIX_SIZE_WIDE;
    m_x_pix_size_um[r][COLSHALF]   = PIX_SIZE_WIDE;
    m_pix_area_arr [r][COLSHALF-1] = arr_wide;
    m_pix_area_arr [r][COLSHALF]   = arr_wide;
  }

  arr_wide = PIX_SIZE_WIDE/PIX_SIZE_ROWS;
  // set size and area for two middle rows of segment
  for(gsize_t c=0; c<COLS; c++) {
    m_y_pix_size_um[ROWSHALF-1][c] = PIX_SIZE_WIDE;
    m_y_pix_size_um[ROWSHALF][c]   = PIX_SIZE_WIDE;
    m_pix_area_arr [ROWSHALF-1][c] = arr_wide;
    m_pix_area_arr [ROWSHALF][c]   = arr_wide;
  }

  // set size and area for four pixels in the center
  pixel_area_t arr_center = PIX_SIZE_WIDE*PIX_SIZE_WIDE / (PIX_SIZE_ROWS*PIX_SIZE_COLS);
    m_pix_area_arr[ROWSHALF-1][COLSHALF-1] = arr_center;
    m_pix_area_arr[ROWSHALF  ][COLSHALF-1] = arr_center;
    m_pix_area_arr[ROWSHALF-1][COLSHALF  ] = arr_center;
    m_pix_area_arr[ROWSHALF  ][COLSHALF  ] = arr_center;

  m_done_bits ^= 2; // set bit 2
}

//--------------

void SegGeometryEpix100V1::print_member_data()
{
  cout << "SegGeometryEpix100V1::print_member_data():"       
       << "\n ROWS                  " << ROWS       
       << "\n COLS                  " << COLS       
       << "\n ROWSHALF              " << ROWSHALF   
       << "\n COLSHALF              " << COLSHALF   
       << "\n PIX_SIZE_COLS         " << PIX_SIZE_COLS 
       << "\n PIX_SIZE_ROWS         " << PIX_SIZE_ROWS 
       << "\n PIX_SIZE_WIDE         " << PIX_SIZE_WIDE    
       << "\n PIX_SIZE_UM           " << PIX_SCALE_SIZE
       << "\n UM_TO_PIX             " << UM_TO_PIX
       << "\n m_use_wide_pix_center " << m_use_wide_pix_center 
       << "\n";
}

//--------------

void SegGeometryEpix100V1::print_coord_arrs()
{
  cout << "\nSegGeometryEpix100V1::print_coord_arrs\n";

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

void SegGeometryEpix100V1::print_min_max_coords()
{
  cout << "  Segment coordinate map limits:"
       << "\n  xmin =  " << pixel_coord_min(AXIS_X)
       << "\n  xmax =  " << pixel_coord_max(AXIS_X)
       << "\n  ymin =  " << pixel_coord_min(AXIS_Y)
       << "\n  ymax =  " << pixel_coord_max(AXIS_Y)
       << "\n  zmin =  " << pixel_coord_min(AXIS_Z)
       << "\n  zmax =  " << pixel_coord_max(AXIS_Z)
       << "\n";
}

//--------------

void SegGeometryEpix100V1::print_seg_info(const unsigned& pbits)
{
  if (pbits & 1) print_member_data();
  if (pbits & 2) print_coord_arrs();
  if (pbits & 4) print_min_max_coords();
}

//--------------

const pixel_area_t* SegGeometryEpix100V1::pixel_area_array()
{
  make_pixel_size_arrs();
  return &m_pix_area_arr[0][0];
}

//--------------

const pixel_coord_t* SegGeometryEpix100V1::pixel_size_array(AXIS axis)
{
  make_pixel_size_arrs();
  if      (axis == AXIS_X) return &m_x_pix_size_um[0][0];
  else if (axis == AXIS_Y) return &m_y_pix_size_um[0][0];
  else if (axis == AXIS_Z) return &m_z_pix_size_um[0][0];
  else                     return &m_y_pix_size_um[0][0];
}

//--------------

const pixel_coord_t* SegGeometryEpix100V1::pixel_coord_array(AXIS axis) 
{ 
  if      (axis == AXIS_X) return &m_x_pix_coord_um[0][0];
  else if (axis == AXIS_Y) return &m_y_pix_coord_um[0][0];
  else if (axis == AXIS_Z) return &m_z_pix_coord_um[0][0];
  else                     return &m_x_pix_coord_um[0][0];
} 

//--------------

const pixel_coord_t SegGeometryEpix100V1::pixel_coord_min(AXIS axis) 
{ 
  const pixel_coord_t* arr = pixel_coord_array (axis);
  pixel_coord_t corner_coords[NCORNERS];
  for (gsize_t i=0; i<NCORNERS; ++i) {corner_coords[i] = arr[IND_CORNER[i]];}
  return geometry::min_of_arr(&corner_coords[0], NCORNERS); 
}

//--------------

const pixel_coord_t SegGeometryEpix100V1::pixel_coord_max(AXIS axis) 
{ 
  const pixel_coord_t* arr = pixel_coord_array (axis);
  pixel_coord_t corner_coords[NCORNERS];
  for (gsize_t i=0; i<NCORNERS; ++i) {corner_coords[i] = arr[IND_CORNER[i]];}
  return geometry::max_of_arr(&corner_coords[0], NCORNERS); 
}

//--------------

const pixel_mask_t* SegGeometryEpix100V1::pixel_mask_array(const unsigned& mbits)
{

  //cout << "SegGeometryEpix100V1::pixel_mask_array(): mbits =" << mbits << '\n';   

  std::fill_n(&m_pix_mask_arr[0][0], int(SIZE), pixel_mask_t(1));

  gsize_t ch = COLSHALF;
  gsize_t rh = ROWSHALF;

  if(mbits & 1) {
    // mask edges
    for (gsize_t r=0; r<ROWS; r++) {
      m_pix_mask_arr[r][0]      = 0;
      m_pix_mask_arr[r][COLS-1] = 0;
    }

    for (gsize_t c=0; c<COLS; c++) {
      m_pix_mask_arr[0][c]      = 0;
      m_pix_mask_arr[ROWS-1][c] = 0;
    }
  } 

  if(mbits & 2) {
    // mask two central columns
    for (gsize_t r=0; r<ROWS; r++) {
      m_pix_mask_arr[r][ch-1] = 0;
      m_pix_mask_arr[r][ch]   = 0;
    }

    // mask two central rows
    for (gsize_t c=0; c<COLS; c++) {
      m_pix_mask_arr[rh-1][c] = 0;
      m_pix_mask_arr[rh][c]   = 0;
    }
  }

  return &m_pix_mask_arr[0][0];
}

//--------------

} // namespace geometry

//--------------
