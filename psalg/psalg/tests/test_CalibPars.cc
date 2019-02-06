// == Build locally
// cd .../lcls2/psalg/build
// make
// == Then run
// psalg/test_CalibPars
// 
// Build in lcls2/install/bin/
// see .../lcls2/psalg/psalg/CMakeLists.txt
// cd .../lcls2
// ./build_all.sh

// Then run it (from lcls2/install/bin/datareader) as 
// test_CalibPars

//#include <iostream> // cout, puts etc.
#include <stdio.h>  // printf

#include "psalg/calib/CalibParsStore.hh"
//#include "psalg/calib/CalibPars.hh"

//using namespace std;
//using namespace psalg;
using namespace calib;
//using namespace detector;

//-------------------

void print_hline(const uint nchars, const char c) {printf("%s\n", std::string(nchars,c).c_str());}

//-------------------
//-------------------
//-------------------
//-------------------

void test_getCalibPars() {
  MSG(INFO, "In test_getCalibPars test access to CalibPars through the factory method getCalibPars");
  query_t query = 123;
  CalibPars* cp = getCalibPars("Epix100a");
  std::cout << "detname: " << cp->detname() << '\n';
  const NDArray<pedestals_t>& peds = cp->pedestals(query);
  const NDArray<common_mode_t>& cmod = cp->common_mode(query);
  std::cout << "\n  peds   : " << peds; 
  std::cout << "\n  cmod   : " << cmod; 
  std::cout << '\n';

  delete cp;
}

//-------------------

void test_CalibPars() {
  MSG(INFO, "In test_CalibPars - test all interface methods");
  CalibPars* cp1 = new CalibPars("Epix100a");
  query_t query = 123;

  std::cout << "detname: " << cp1->detname() << '\n';

  const NDArray<pedestals_t>&    peds   = cp1->pedestals       (query);
  const NDArray<common_mode_t>&  cmod   = cp1->common_mode     (query);
  const NDArray<pixel_rms_t>&    rms    = cp1->rms             (query);
  const NDArray<pixel_status_t>& stat   = cp1->status          (query);
  const NDArray<pixel_gain_t>&   gain   = cp1->gain            (query);
  const NDArray<pixel_offset_t>& offset = cp1->offset          (query);
  const NDArray<pixel_bkgd_t>&   bkgd   = cp1->background      (query);
  const NDArray<pixel_mask_t>&   maskc  = cp1->mask_calib      (query);
  const NDArray<pixel_mask_t>&   masks  = cp1->mask_from_status(query);
  const NDArray<pixel_mask_t>&   maske  = cp1->mask_edges      (query, 8);
  const NDArray<pixel_mask_t>&   maskn  = cp1->mask_neighbors  (query, 1, 1);
  const NDArray<pixel_mask_t>&   mask2  = cp1->mask_bits       (query, 0177777);
  const NDArray<pixel_mask_t>&   mask3  = cp1->mask            (query, true, true, true, true);
  const NDArray<pixel_idx_t>&    idxx   = cp1->indexes         (query, 0);
  const NDArray<pixel_coord_t>&  coordsx= cp1->coords          (query, 0);
  const NDArray<pixel_size_t>&   sizex  = cp1->pixel_size      (query, 0);
  const NDArray<pixel_size_t>&   axisx  = cp1->image_xaxis     (query);
  const NDArray<pixel_size_t>&   axisy  = cp1->image_yaxis     (query);
  const geometry_t&              geotxt = cp1->geometry        (query);

  std::cout << "\n  peds   : " << peds; 
  std::cout << "\n  cmod   : " << cmod; 
  std::cout << "\n  rms    : " << rms; 
  std::cout << "\n  stat   : " << stat; 
  std::cout << "\n  gain   : " << gain; 
  std::cout << "\n  offset : " << offset; 
  std::cout << "\n  bkgd   : " << bkgd; 
  std::cout << "\n  maskc  : " << maskc; 
  std::cout << "\n  masks  : " << masks; 
  std::cout << "\n  maske  : " << maske; 
  std::cout << "\n  maskn  : " << maskn; 
  std::cout << "\n  mask2  : " << mask2; 
  std::cout << "\n  mask3  : " << mask3; 
  std::cout << "\n  idxx   : " << idxx; 
  std::cout << "\n  coordsx: " << coordsx; 
  std::cout << "\n  sizex  : " << sizex; 
  std::cout << "\n  axisx  : " << axisx; 
  std::cout << "\n  axisy  : " << axisy; 
  std::cout << "\n  geotxt : " << geotxt; 
  std::cout << '\n';

  //delete cp1; // !!!! Segmentation fault
}

//-------------------

std::string usage(const std::string& tname="")
{
  std::stringstream ss;
  if (tname == "") ss << "Usage command> test_CalibPars <test-number>\n  where test-number";
  if (tname == "" || tname=="0"	) ss << "\n  0  - test_CalibPars()";
  if (tname == "" || tname=="1"	) ss << "\n  1  - test_getCalibPars()";
  ss << '\n';
  return ss.str();
}

//-------------------

int main(int argc, char **argv) {

  print_hline(80,'_');
  LOGGER.setLogger(LL::DEBUG, "%H:%M:%S.%f");
  MSG(INFO, "In test_CalibPars");

  cout << usage(); 
  print_hline(80,'_');
  if (argc==1) {return 0;}
  std::string tname(argv[1]);
  cout << usage(tname); 

  if      (tname=="0") test_CalibPars();
  else if (tname=="1") test_getCalibPars();
  else MSG(WARNING, "Undefined test name: " << tname);

  print_hline(80,'_');
  return EXIT_SUCCESS;
}

//-------------------
