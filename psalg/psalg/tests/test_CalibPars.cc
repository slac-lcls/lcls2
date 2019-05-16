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

#include "psalg/calib/Response.hh"
#include "psalg/calib/CalibParsDB.hh"
#include "psalg/calib/CalibParsDBStore.hh"

#include "psalg/calib/CalibParsStore.hh"
#include "psalg/calib/CalibPars.hh"

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

void test_CalibPars(const char* detname = "epix100a") { //"undefined"
  MSG(INFO, "In test_CalibPars - test all interface methods");
  CalibPars* cp1 = new CalibPars(detname);
  std::cout << "detname: " << cp1->detname() << '\n';

  Query query("some-string is here");

  const NDArray<pedestals_t>&    peds   = cp1->pedestals       (query);
  const NDArray<common_mode_t>&  cmod   = cp1->common_mode     (query);
  const NDArray<pixel_rms_t>&    rms    = cp1->rms             (query);
  const NDArray<pixel_status_t>& stat   = cp1->status          (query);
  const NDArray<pixel_gain_t>&   gain   = cp1->gain            (query);
  const NDArray<pixel_offset_t>& offset = cp1->offset          (query);
  const NDArray<pixel_bkgd_t>&   bkgd   = cp1->background      (query);
  const NDArray<pixel_mask_t>&   maskc  = cp1->mask_calib      (query);
  const NDArray<pixel_mask_t>&   masks  = cp1->mask_from_status(query);
  const NDArray<pixel_mask_t>&   maske  = cp1->mask_edges      (query);
  const NDArray<pixel_mask_t>&   maskn  = cp1->mask_neighbors  (query);
  const NDArray<pixel_mask_t>&   mask2  = cp1->mask_bits       (query);
  const NDArray<pixel_mask_t>&   mask3  = cp1->mask            (query);
  const NDArray<pixel_idx_t>&    idxx   = cp1->indexes         (query);
  const NDArray<pixel_coord_t>&  coordsx= cp1->coords          (query);
  const NDArray<pixel_size_t>&   sizex  = cp1->pixel_size      (query);
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

void test_getCalibPars(const char* detname = "undefined") { //"epix100a"
  MSG(INFO, "In test_getCalibPars test access to CalibPars through the factory method getCalibPars");

  CalibPars* cp = getCalibPars(detname);
  std::cout << "detname: " << cp->detname() << '\n';

  Query q("some-string is here");
  const NDArray<pedestals_t>&   peds = cp->pedestals(q);
  const NDArray<common_mode_t>& cmod = cp->common_mode(q);
  std::cout << "\n  peds   : " << peds; 
  std::cout << "\n  cmod   : " << cmod; 
  std::cout << '\n';

  delete cp;
}

//-------------------

void test_CalibParsDB(const char* dbtypename = "Default-Base-NoDB") {
  MSG(INFO, "In test_CalibParsDB test access to CalibParsDB");
  Query q("some-string is here");

  CalibParsDB* o0 = new CalibParsDB();
  std::cout << "In test_CalibParsDB dbtypename: " << o0->dbtypename() << '\n';

  CalibParsDB* o = new CalibParsDB(dbtypename);
  std::cout << "In test_CalibParsDB dbtypename: " << o->dbtypename() << '\n';

  const NDArray<float>&  nda_float = o->get_ndarray_float(q);
  std::cout << "\n  nda_float  : " << nda_float; 
  std::cout << "\n  nda_double : " << o->get_ndarray_double(q);
  std::cout << "\n  nda_uint16 : " << o->get_ndarray_uint16(q); 
  std::cout << "\n  nda_uint32 : " << o->get_ndarray_uint32(q); 
  std::cout << "\n  string     : " << o->get_string(q);
  std::cout << '\n';

  delete o0;
  delete o;
}

//-------------------
//-------------------

void test_getCalibParsDB(const DBTYPE& dbtype=DBDEF) {
  MSG(INFO, "In test_getCalibParsDB test access to CalibParsDB through the factory method getCalibParsDB");
  Query q("some-string is here");
  CalibParsDB* o = getCalibParsDB(dbtype);
  std::cout << "In test_CalibParsDB dbtypename: " << o->dbtypename() << '\n';

  /*
  const NDArray<pedestals_t>& peds = cp->pedestals(query);
  const NDArray<common_mode_t>& cmod = cp->common_mode(query);
  std::cout << "\n  peds   : " << peds; 
  std::cout << "\n  cmod   : " << cmod; 
  std::cout << '\n';
  */

  delete o;
}

//-------------------

void test_Query() {
  MSG(INFO, "In test_Query test access to Query");

  Query q1(std::string("query-string is here"));
  std::cout << "q1.query() " << q1.query() << "\n\n";

  //----
  const char* det         = "cspad_0001";
  const char* exp         = "cxid9114";
  const char* ctype       = "pedestals";
  const unsigned run      = 150;
  const unsigned time_sec = 0;  // default
  const char* version     = ""; // default
  Query q2(det, exp, ctype, run, time_sec, version);
  std::cout << "q2: " << q2.string_members("  ") << "\n\n";

  //----
  Query::map_t map = 
    {{Query::DETECTOR,"cspad_0001"}
    ,{Query::EXPERIMENT,"cxid9114"}
    ,{Query::CALIBTYPE,"pedestals"}
    ,{Query::RUN,"150"}
    ,{Query::TIME_SEC,"0"}
    ,{Query::VERSION,""}
    };
  Query q3(map);
  std::cout << "q3: " << q3.string_members("  ") << "\n\n";
  //----

  q3.set_paremeter(Query::RUN,"261");
  q3.set_paremeter(Query::EXPERIMENT,"xyz12345");
  std::cout << "q3.set_paremeter: " << q3.string_members("  ") << "\n\n";


}

//-------------------

void test_Response() {
  MSG(INFO, "In test_Response test access to Response");
  Response r("detector-name-is-here");
  std::cout << "r.detname() " << r.detname() << '\n';
}

//-------------------

std::string usage(const std::string& tname="")
{
  std::stringstream ss;
  if (tname == "") ss << "Usage command> test_CalibPars <test-number>\n  where test-number";
  if (tname == "" || tname=="0"	) ss << "\n   0  - test_CalibPars()    - basee class";
  if (tname == "" || tname=="1"	) ss << "\n   1  - test_getCalibPars() - default for undefined";
  if (tname == "" || tname=="2"	) ss << "\n   2  - test_getCalibPars('epix100a') - default - the same as base";
  if (tname == "" || tname=="10") ss << "\n  10  - test_CalibParsDB()";
  if (tname == "" || tname=="11") ss << "\n  11  - test_getCalibParsDB(DBDEF)";
  if (tname == "" || tname=="12") ss << "\n  12  - test_getCalibParsDB(DBWEB)";
  if (tname == "" || tname=="20") ss << "\n  20  - test_Query()";
  if (tname == "" || tname=="30") ss << "\n  30  - test_Response()";
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

  if      (tname=="0")  test_CalibPars();
  else if (tname=="1")  test_getCalibPars(); // "undefined"
  else if (tname=="2")  test_getCalibPars("epix100a");

  else if (tname=="10") test_CalibParsDB();
  else if (tname=="11") test_getCalibParsDB(DBDEF);
  else if (tname=="12") test_getCalibParsDB(DBWEB);

  else if (tname=="20") test_Query();
  else if (tname=="30") test_Response();
  else MSG(WARNING, "Undefined test name: " << tname);

  print_hline(80,'_');
  return EXIT_SUCCESS;
}

//-------------------
