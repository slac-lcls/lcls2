// == Build locally
// cd .../lcls2/psalg/build
// make
// == Then run
// psalg/test_AreaDetector
// 
// Build in lcls2/install/bin/
// see .../lcls2/psalg/psalg/CMakeLists.txt
// cd .../lcls2
// ./build_all.sh

// Then run it (from lcls2/install/bin/datareader) as 
// test_AreaDetector

//#include <iostream> // cout, puts etc.
#include <stdio.h>  // printf

#include "psalg/detector/AreaDetectorStore.hh"
//#include "psalg/detector/AreaDetector.hh"

//using namespace std;
//using namespace psalg;
using namespace calib;
using namespace detector;

//-------------------

void print_hline(const uint nchars, const char c) {printf("%s\n", std::string(nchars,c).c_str());}

//-------------------
//-------------------
//-------------------
//-------------------

void test_AreaDetector() {
  MSG(INFO, "In test_AreaDetector - test base class interface methods");
  query_t query = 123;
  AreaDetector* det = new AreaDetector("Epix100a");
  std::cout << "detname: " << det->detname() << '\n';

  const NDArray<pedestals_t>& peds = det->pedestals(query);
  const NDArray<common_mode_t>& cmod = det->common_mode(query);
  const NDArray<raw_t>& raw = det->raw(query);
  std::cout << "\n  peds   : " << peds;
  std::cout << "\n  cmod   : " << cmod;
  std::cout << "\n  raw    : " << raw;
  std::cout << '\n';

  delete det;
}

//-------------------

void test_getAreaDetector() {
  MSG(INFO, "In test_getAreaDetector test factory method getAreaDetector");
  query_t query = 123;

  AreaDetector* det = getAreaDetector("Epix100a");
  std::cout << "detname: " << det->detname() << '\n';

  const NDArray<pedestals_t>&    peds   = det->pedestals       (query);
  const NDArray<common_mode_t>&  cmod   = det->common_mode     (query);
  const NDArray<pixel_rms_t>&    rms    = det->rms             (query);
  const NDArray<pixel_status_t>& stat   = det->status          (query);
  const NDArray<pixel_gain_t>&   gain   = det->gain            (query);
  const NDArray<pixel_offset_t>& offset = det->offset          (query);
  const NDArray<pixel_bkgd_t>&   bkgd   = det->background      (query);
  const NDArray<pixel_mask_t>&   maskc  = det->mask_calib      (query);
  const NDArray<pixel_mask_t>&   masks  = det->mask_from_status(query);
  const NDArray<pixel_mask_t>&   maske  = det->mask_edges      (query, 8);
  const NDArray<pixel_mask_t>&   maskn  = det->mask_neighbors  (query, 1, 1);
  const NDArray<pixel_mask_t>&   mask2  = det->mask_bits       (query, 0177777);
  const NDArray<pixel_mask_t>&   mask3  = det->mask            (query, true, true, true, true);
  const NDArray<pixel_idx_t>&    idxx   = det->indexes         (query, 0);
  const NDArray<pixel_coord_t>&  coordsx= det->coords          (query, 0);
  const NDArray<pixel_size_t>&   sizex  = det->pixel_size      (query, 0);
  const NDArray<pixel_size_t>&   axisx  = det->image_xaxis     (query);
  const NDArray<pixel_size_t>&   axisy  = det->image_yaxis     (query);
  const geometry_t&              geotxt = det->geometry        (query);
  const NDArray<raw_t>&          raw    = det->raw             (query);
  const NDArray<calib_t>&        calib  = det->calib           (query);
  const NDArray<image_t>&        image  = det->image           (query);

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
  std::cout << "\n  raw    : " << raw;
  std::cout << "\n  calib  : " << calib;
  std::cout << "\n  image  : " << image;
  std::cout << '\n';

  delete det;
 }

//-------------------

int main(int argc, char **argv) {

  MSG(INFO, LOGGER.tstampStart() << " Logger started"); // optional record
  LOGGER.setLogger(LL::DEBUG, "%H:%M:%S.%f");           // set level and time format

  print_hline(80,'_');
  test_AreaDetector();    print_hline(80,'_');
  test_getAreaDetector(); print_hline(80,'_');
  return EXIT_SUCCESS;
}

//-------------------
