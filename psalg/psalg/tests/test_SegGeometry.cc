// == Build locally
// cd .../lcls2/psalg/build
// make
// == Then run
// psalg/test_SegGeometry
// 
// Build in lcls2/install/bin/
// see .../lcls2/psalg/psalg/CMakeLists.txt
// cd .../lcls2
// ./build_all.sh

// Then run it (from lcls2/install/bin/datareader) as 
// test_SegGeometry

#include <iostream> // cout, puts etc.
#include <stdio.h>  // printf

#include "psalg/geometry/SegGeometryStore.hh"

//using namespace std;
//using namespace calib;
//using namespace detector;
using namespace psalg;

//typedef psalg::SegGeometry SG;

//-------------------

void print_hline(const uint nchars, const char c) {printf("%s\n", std::string(nchars,c).c_str());}

//-------------------

void test_SegGeometry(const char* segname = "SENS2X1:V1") { //"undefined"
  MSG(INFO, "In test_SegGeometry - test_SegGeometry(" << segname << ")");

  SegGeometry *seggeom = SegGeometryStore::Create(segname, 0377);  
  seggeom -> print_seg_info(0377);
}

//-------------------

std::string usage(const std::string& tname="")
{
  std::stringstream ss;
  if (tname == "") ss << "Usage command> test_SegGeometry <test-number>\n  where test-number";
  if (tname == "" || tname=="0"	) ss << "\n   0  - test_SegGeometry()           ";
  if (tname == "" || tname=="1"	) ss << "\n   1  - test_SegGeometry(\"SENS2X1:V1\") ";
  if (tname == "" || tname=="2"	) ss << "\n   2  - test_SegGeometry(\"EPIX100:V1\") ";
  if (tname == "" || tname=="3"	) ss << "\n   3  - test_SegGeometry(\"EPIX10KA:V1\")";
  if (tname == "" || tname=="4"	) ss << "\n   4  - test_SegGeometry(\"PNCCD:V1\")   ";
  if (tname == "" || tname=="5"	) ss << "\n   5  - test_SegGeometry(\"MTRX:384:384:100:100\")";
  ss << '\n';
  return ss.str();
}

//-------------------

int main(int argc, char **argv) {

  print_hline(80,'_');
  LOGGER.setLogger(LL::DEBUG, "%H:%M:%S.%f");
  MSG(INFO, "In test_SegGeometry");

  cout << usage(); 
  print_hline(80,'_');
  if (argc==1) {return 0;}
  std::string tname(argv[1]);
  cout << usage(tname); 

  if      (tname=="0")  test_SegGeometry();
  else if (tname=="1")  test_SegGeometry("SENS2X1:V1");
  else if (tname=="2")  test_SegGeometry("EPIX100:V1");
  else if (tname=="3")  test_SegGeometry("EPIX10KA:V1");
  else if (tname=="4")  test_SegGeometry("PNCCD:V1");
  else if (tname=="5")  test_SegGeometry("MTRX:384:384:100:100");       

  else MSG(WARNING, "Undefined test name: " << tname);

  print_hline(80,'_');
  return EXIT_SUCCESS;
}

//-------------------
