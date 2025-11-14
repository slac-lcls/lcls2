/* 
 * https://curl.haxx.se/libcurl/c/getinfo.html
 */

#include "psalg/utils/Logger.hh" // MSG, LOGGER,...

#include <stdio.h>
#include <iostream> // cout
#include <sstream>
//#include <vector>

#include "psalg/calib/NDArray.hh"
#include "psana/peakFinder/include/WFAlgos.hh"

using namespace psalg;

//-------------------

void print_hline(const uint nchars, const char c) {printf("%s\n", std::string(nchars,c).c_str());}

//-------------------

const double WFTESTDATA[] = {
    1,
   -1,
    0,
    1,
   -2,
   -5,
  -15,
  -85,
 -100,
 -101,
 -102,
 -103,
 -104,
 -102,
 -100,
  -55,
  -35,
  -25,
  -20,
  -15,
  -10,
   -9,
   -5,
    1,
   -1,
    0,
    1,
   -2,
   -5,
  -15,
  -85,
 -100,
 -101,
 -102,
 -103,
 -105,
 -102,
 -100,
  -55,
  -35,
  -25,
  -20,
  -15,
  -10,
   -9,
   -5,
    1,
   -1,
    0,
    1,
   -1,
    0,
    1,
   -1,
    0,
    1,
    1,
   -1,
    0,
    1,
    1,
    0,
    1,
    1,
};
 
//-------------------

void test_find_edges() {
  printf("In test_WFAlgos::test_find_edges\n");

  uint32_t n = sizeof(WFTESTDATA)/sizeof(WFTESTDATA[0]);
  std::vector<double> waveform(WFTESTDATA, WFTESTDATA+n);

  double   baseline = 0;
  double   threshold=-5;
  double   fraction = 0.5;
  double   deadtime = 0;
  bool     leading_edges=true;
  uint32_t npkmax = 10;
  double   pkvals[10];
  uint32_t pkinds[10];

  uint32_t npks = find_edges(npkmax, pkvals, pkinds, waveform, baseline, threshold, fraction, deadtime, leading_edges);

  std::cout << "\nIn test_find_edges - find_edges found npk: " << npks << '\n'; 
  std::cout << "\n  pkinds: ";
  for(index_t i=0; i<npks; ++i) std::cout << pkinds[i] << ' ';
  std::cout << "\n  pkvals: ";
  for(index_t i=0; i<npks; ++i) std::cout << pkvals[i] << ' ';
  std::cout << '\n';
}

//-------------------
 
void test_input_pars(int i, const int j, int& k, const int& l) {
  cout << "\nvalue received as       int : " << i; 
  cout << "\nvalue received as const int : " << j; 
  cout << "\nvalue received as       int&: " << k; 
  cout << "\nvalue received as const int&: " << l; 
  cout << '\n'; 
}

void test_cpp() {
  printf("In test_cpp\n");
  const int i=1; const int j=2; int k=3; const int l=4;
  test_input_pars(i,j,k,l);
}

//-------------------

std::string usage(const std::string& tname="")
{
  std::stringstream ss;
  if (tname == "") ss << "Usage command> test_WFAlgos <test-number>\n  where test-number";
  if (tname == "" || tname=="1"	) ss << "\n  1  - test_cpp()";
  if (tname == "" || tname=="2"	) ss << "\n  2  - test_find_edges() - uses vectors for IO";
  ss << '\n';
  return ss.str();
}

//-------------------

int main(int argc, char* argv[])
{
  print_hline(80,'_');
  //MSG(INFO, LOGGER.tstampStart() << " Logger started"); // optional record
  LOGGER.setLogger(LL::DEBUG, "%H:%M:%S.%f");           // set level and time format
  MSG(INFO, "In test_WFAlgos");

  print_hline(80,'_');
  cout << usage();
  if (argc==1) {return 0;}
  std::string tname(argv[1]);
  cout << usage(tname); 

  if      (tname=="1")  test_cpp();
  else if (tname=="2")  test_find_edges();
  else MSG(WARNING, "Undefined test name \"" << tname << '\"');
 
  print_hline(80,'_');
  return EXIT_SUCCESS;
}

//-------------------

