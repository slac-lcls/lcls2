//-------------------

#include <time.h>   // time
#include <string>
#include <iostream>
#include <iomanip>  // for setw, setfill

//#include "psalg/geometry/GeometryObject.hh"
#include "psalg/geometry/GeometryAccess.hh"
#include "psalg/calib/NDArray.hh"
#include "psalg/utils/MacTimeFix.hh"

using namespace std;
//using namespace geometry;

typedef geometry::GeometryObject::SG SG;

struct timespec start, stop;
int status;

//-------------------

string fname_cspad2x2("/reg/g/psdm/detector/alignment/cspad2x2/calib-cspad2x2-01-2013-02-13/calib/"
                      "CsPad2x2::CalibV1/MecTargetChamber.0:Cspad2x2.1/geometry/0-end.data");

string fname_cspad("/reg/g/psdm/detector/alignment/cspad/calib-mec-2017-10-20/calib/"
	           "CsPad::CalibV1/MecTargetChamber.0:Cspad.0/geometry/0-end.data");

//-------------------

double time_sec_nsec(const timespec& t){return t.tv_sec + 1e-9*(t.tv_nsec);}

//-------------------

double dtime(const timespec& start, const timespec& stop){
  return time_sec_nsec(stop) - time_sec_nsec(start);
}

//-------------------

void test_file_to_stringstream()
{
  const string& fname = fname_cspad2x2;
  cout << "\n==test_file_to_stringstream fname: " << fname << " \n";

  std::stringstream ss;
  geometry::file_to_stringstream(fname, ss);
  cout << "string:\n" << ss.str() <<  "\n";
}

//-------------------

void test_geometry()
{
  const string& fname = fname_cspad;
  cout << "\n==test_geometry fname: " << fname << " \n";

  geometry::GeometryAccess geo(fname);

  //geo.print_comments_from_dict();
  //geo.print_list_of_geos();
  //geo.print_list_of_geos_children();
  geo.print_geometry_info(7);

  status = clock_gettime(CLOCK_REALTIME, &start);
  geo.print_pixel_coords();
  status = clock_gettime(CLOCK_REALTIME, &stop);

  cout << "\nconsumed time = " << dtime(start, stop) << " sec\n";
}

//-------------------

void test_geo_get_pixel_coords_as_pointer()
{
  LOGGER.setLogger(LL::INFO, "%H:%M:%S");

  const string& fname = fname_cspad;
  cout << "\n==test_geo_get_pixel_coords_as_pointer fname: " << fname << " \n";

  geometry::GeometryAccess geo(fname);
  const double* X;
  const double* Y;
  const double* Z;
  unsigned   size;
  geo.get_pixel_coords(X,Y,Z,size); //,oname,oindex,do_tilt

  std::stringstream ss; ss << "print_pixel_coords():\n"
			   << "size=" << size << '\n' << std::fixed << std::setprecision(1);  
  ss << "X: "; for(unsigned i=0; i<10; ++i) ss << std::setw(10) << X[i] << ", "; ss << "...\n";
  ss << "Y: "; for(unsigned i=0; i<10; ++i) ss << std::setw(10) << Y[i] << ", "; ss << "...\n"; 
  ss << "Z: "; for(unsigned i=0; i<10; ++i) ss << std::setw(10) << Z[i] << ", "; ss << "...\n"; 
  //cout << ss.str();
  MSG(INFO, ss.str());
}

//-------------------

void test_geo_get_pixel_coords_as_ndarray()
{
  LOGGER.setLogger(LL::INFO, "%H:%M:%S");

  const string& fname = fname_cspad;
  cout << "\n==test_geo_get_pixel_coords_as_ndarray fname: " << fname << " \n";

  geometry::GeometryAccess geo(fname);

  psalg::NDArray<const double>* pxarr = geo.get_pixel_coords(SG::AXIS_X);
  psalg::NDArray<const double>* pyarr = geo.get_pixel_coords(SG::AXIS_Y);
  psalg::NDArray<const double>* pzarr = geo.get_pixel_coords(SG::AXIS_Z);

  cout << "  X: " << *pxarr << '\n';
  cout << "  Y: " << *pyarr << '\n';
  cout << "  Z: " << *pzarr << '\n';
}

//-------------------

void test_geo_get_misc()
{
  LOGGER.setLogger(LL::INFO, "%H:%M:%S");

  const string& fname = fname_cspad;
  cout << "\n==test_geo_get_misc fname: " << fname << " \n";

  geometry::GeometryAccess geo(fname);

  const double pix_scale_size_um = 109.92;
  const int xy0_off_pix_v2[] = {500,500};
  const int xy0_off_pix_v1[] = {300,300};
  const double Zplane1 = 1000000; //[um] or 0
  const double Zplane2 = 100000; //[um] or 0

  psalg::NDArray<const double>*   areas = geo.get_pixel_areas();
  psalg::NDArray<const int>*      mask  = geo.get_pixel_mask(0377);
  psalg::NDArray<const double>*   xatz  = geo.get_pixel_coords_at_z(Zplane1, SG::AXIS_X);
  psalg::NDArray<const unsigned>* ix    = geo.get_pixel_coord_indexes(SG::AXIS_X, pix_scale_size_um, xy0_off_pix_v1);
  psalg::NDArray<const unsigned>* ixatz = geo.get_pixel_inds_at_z(Zplane2, SG::AXIS_X, pix_scale_size_um, xy0_off_pix_v2);

  cout << "  areas: " << *areas << '\n';
  cout << "  mask : " << *mask  << '\n';
  cout << "  xatz : " << *xatz  << '\n';
  cout << "  ix   : " << *ix    << '\n';
  cout << "  ixatz: " << *ixatz << '\n';
  //cout << "   : " << * << '\n';
}

//-------------------

void print_hline(const unsigned nchars, const char c) {printf("%s\n", std::string(nchars,c).c_str());}

//-------------------

std::string usage(const std::string& tname="")
{
  std::stringstream ss;
  if (tname == "") ss << "Usage command> test_GeometryAccess <test-number>\n  where test-number";
  if (tname == "" || tname=="0"	) ss << "\n   0  - test_file_to_stringstream()";
  if (tname == "" || tname=="1"	) ss << "\n   1  - test_geometry()";
  if (tname == "" || tname=="2"	) ss << "\n   2  - test_geo_get_pixel_coords_as_pointer()";
  if (tname == "" || tname=="3"	) ss << "\n   3  - test_geo_get_pixel_coords_as_ndarray()";
  if (tname == "" || tname=="4"	) ss << "\n   4  - test_geo_get_misc()";
  ss << '\n';
  return ss.str();
}

//-------------------

int main(int argc, char* argv[])
{
  MSG(INFO, LOGGER.tstampStart() << " Logger started"); // optional record
  //LOGGER.setLogger(LL::INFO, "%H:%M:%S");           // "%H:%M:%S.%f"
  LOGGER.setLogger(LL::DEBUG, "%H:%M:%S");

  cout << usage(); 
  print_hline(80,'_');
  if (argc==1) {return 0;}
  std::string tname(argv[1]);
  cout << usage(tname); 

  if      (tname=="0")  test_file_to_stringstream();
  else if (tname=="1")  test_geometry();
  else if (tname=="2")  test_geo_get_pixel_coords_as_pointer();
  else if (tname=="3")  test_geo_get_pixel_coords_as_ndarray();
  else if (tname=="4")  test_geo_get_misc();

  else MSG(WARNING, "Undefined test name: " << tname);

  print_hline(80,'_');
  return EXIT_SUCCESS;
}

//-----------------
