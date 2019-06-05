//-------------------

#include "psalg/geometry/GeometryObject.hh"
#include "psalg/geometry/GeometryAccess.hh"
 
#include <string>
#include <iostream>
#include <iomanip>  // for setw, setfill

using namespace std;
//using namespace geometry;

//-------------------

void test_file_to_stringstream()
{
  string fname("/reg/g/psdm/detector/alignment/cspad2x2/calib-cspad2x2-01-2013-02-13/calib/"
               "CsPad2x2::CalibV1/MecTargetChamber.0:Cspad2x2.1/geometry/0-end.data");
  cout << "\ntest_file_to_stringstream fname: " << fname << " \n";

  std::stringstream ss;
  geometry::file_to_stringstream(fname, ss);
  cout << "string:\n" << ss.str() <<  "\n";
}

//-------------------

void test_geometry()
{
  string fname("/reg/g/psdm/detector/alignment/cspad/calib-mec-2017-10-20/calib/"
	       "CsPad::CalibV1/MecTargetChamber.0:Cspad.0/geometry/0-end.data");
  cout << "\ntest_geometry_loading fname: " << fname << " \n";

  geometry::GeometryAccess geo(fname);

  geo.print_comments_from_dict();
  geo.print_list_of_geos();
  geo.print_list_of_geos_children();
  geo.print_pixel_coords();
}

//-------------------

void print_hline(const uint nchars, const char c) {printf("%s\n", std::string(nchars,c).c_str());}

//-------------------

std::string usage(const std::string& tname="")
{
  std::stringstream ss;
  if (tname == "") ss << "Usage command> test_GeometryAccess <test-number>\n  where test-number";
  if (tname == "" || tname=="0"	) ss << "\n   0  - test_file_to_stringstream()";
  if (tname == "" || tname=="1"	) ss << "\n   1  - test_geometry()";
  ss << '\n';
  return ss.str();
}

//-------------------

int main(int argc, char* argv[])
{
  MSG(INFO, LOGGER.tstampStart() << " Logger started"); // optional record
  LOGGER.setLogger(LL::INFO, "%H:%M:%S");           // "%H:%M:%S.%f"
  //LOGGER.setLogger(LL::DEBUG, "%H:%M:%S");

  cout << usage(); 
  print_hline(80,'_');
  if (argc==1) {return 0;}
  std::string tname(argv[1]);
  cout << usage(tname); 

  if      (tname=="0")  test_file_to_stringstream();
  else if (tname=="1")  test_geometry();

  else MSG(WARNING, "Undefined test name: " << tname);

  print_hline(80,'_');
  return EXIT_SUCCESS;
}

//-----------------
