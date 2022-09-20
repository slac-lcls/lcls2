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

#include <fcntl.h> // O_RDONLY
#include <stdio.h> // for  sprintf, printf( "%lf\n", accum );
#include <iostream> // for cout, puts etc.
#include <unistd.h> // close
#include <stdint.h>  // uint8_t, uint32_t, etc.

#include <bitset>

//#include "xtcdata/xtc/ShapesData.hh"
//#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/XtcFileIterator.hh"
#include "xtcdata/xtc/ConfigIter.hh" // < NamesIter < XtcIterator
#include "xtcdata/xtc/DataIter.hh" //  < XtcIterator
#include "xtcdata/xtc/NamesLookup.hh"

#include "psalg/detector/AreaDetectorStore.hh" // AreaDetectorJungfrau
#include "psalg/detector/AreaDetector.hh"
#include "psalg/detector/UtilsConfig.hh" // print_dg_info

#include "psalg/detector/AreaDetectorJungfrau.hh"
#include "psalg/detector/AreaDetectorPnccd.hh"
#include "psalg/detector/AreaDetectorOpal.hh"
//#include "psalg/detector/AreaDetectorCspad.hh"

using namespace calib;
using namespace detector;

using namespace std;
using namespace XtcData;

//-------------------

void print_hline(const unsigned nchars, const char c) {printf("%s\n", std::string(nchars,c).c_str());}

//-------------------
//-------------------
//-------------------
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
  NDArray<const pixel_idx_t>&    idxx   = det->indexes         (query, 0);
  NDArray<const pixel_coord_t>&  coordsx= det->coords          (query, 0);
  NDArray<const pixel_size_t>&   sizex  = det->pixel_size      (query, 0);
  NDArray<const pixel_size_t>&   axisx  = det->image_xaxis     (query);
  NDArray<const pixel_size_t>&   axisy  = det->image_yaxis     (query);
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

int file_descriptor(const char* fname="data-xpptut15-r0430-e000010-jungfrau.xtc2",
                    const char* dname="/reg/g/psdm/detector/data2_test/xtc/") {
  //const char* path = "/reg/neh/home/cpo/git/lcls2/psana/psana/dgramPort/jungfrau.xtc2";
  //const char* path = "/reg/g/psdm/detector/data2_test/xtc/data-cxid9114-r0089-e000010-cspad.xtc2";
  //const char* path = "/reg/g/psdm/detector/data2_test/xtc/data-xpptut15-r0430-e000010-jungfrau.xtc2";
    string path(dname); path += fname;
    std::cout << "xtc file name: " << path.c_str() << '\n';
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Unable to open file '%s'\n", path.c_str());
        exit(2);
    }
    return fd;
}

//-------------------

void test_AreaDetector_base(int argc, char* argv[]) {
  MSG(INFO, "In test_AreaDetector_base");

  int fd = file_descriptor("data-xpptut15-r0430-e000010-jungfrau.xtc2");
  XtcFileIterator xfi(fd, 0x4000000);

  // get configuration out of the 1-st datagram
  Dgram* dg = xfi.next();
  const void* bufEnd = ((char*)dg) + xfi.size();
  ConfigIter ci(&(dg->xtc), bufEnd);

  //AreaDetectorJungfrau& det = (AreaDetectorJungfrau&)*getAreaDetector("jungfrau", ci);
  AreaDetector& det = *getAreaDetector("jungfrau", ci);

  det.AreaDetector::process_config();
  //det.process_config();

  unsigned neventreq=2;
  unsigned nevent=0;
  while ((dg = xfi.next())) {
      if (nevent>=neventreq) break;

      printf("evt:%04d\n", nevent);
      print_dg_info(dg);

      DataIter di(&(dg->xtc), bufEnd);

      det.AreaDetector::process_data(di);
      //det.process_data(di);

      nevent++;
  }
  ::close(fd);
}

//-------------------
//-------------------

void test_AreaDetector_mix(int argc, char* argv[]) {
  MSG(INFO, "In test_AreaDetector_mix");

  int fd = file_descriptor("data-xpptut15-r0430-e000010-jungfrau.xtc2");
  XtcFileIterator xfi(fd, 0x4000000);

  Dgram* dg = xfi.next();
  const void* bufEnd = ((char*)dg) + xfi.size();
  ConfigIter ci(&(dg->xtc), bufEnd);

  //AreaDetectorJungfrau& det = (AreaDetectorJungfrau&)*getAreaDetector("jungfrau", ci);
  AreaDetector& det = *getAreaDetector("jungfrau", ci);
  //AreaDetector det("jungfrau", ci);

  //det.AreaDetector::process_config();
  //det.process_config();

  det.print_config();
  std::cout << str_config_names(ci) << '\n';

  //=======
  //query_t query = 123;
  //const NDArray<pedestals_t>& peds = det->pedestals(query);
  //const NDArray<common_mode_t>& cmod = det->common_mode(query);
  //const NDArray<raw_t>& raw = det->raw(query);
  //std::cout << "\n  peds   : " << peds;
  //std::cout << "\n  cmod   : " << cmod;
  //std::cout << "\n  raw    : " << raw;
  //std::cout << '\n';
  //NamesLookup& namesLookup = ci.namesLookup();
  //=======

  unsigned neventreq=2;
  unsigned nevent=0;
  while ((dg = xfi.next())) {
      if (nevent>=neventreq) break;

      printf("evt:%04d\n", nevent);
      print_dg_info(dg);

      DataIter di(&(dg->xtc), bufEnd);

      //DESC_VALUE(desc_data, di, namesLookup);
      //DescData& desc_data = di.desc_value(namesLookup);
      //dump("Data values", desc_data);

      //det.AreaDetector::process_data(di);
      det.process_data(di); // <- prints data details

      raw_t* data;
      det.raw<raw_t>(di, data, "frame");
      printf(" == raw data pointer: %d %d %d %d %d\n", data[0],data[1],data[2],data[3],data[4]);

      NDArray<raw_t> nda;
      det.raw<raw_t>(di, nda);
      std::cout << " == raw data nda: " << nda << '\n';

      nevent++;
  }
  ::close(fd);
}

//-------------------

void test_AreaDetector(int argc, char* argv[]) {
  MSG(INFO, "In test_AreaDetector");

  int fd = file_descriptor("data-xpptut15-r0430-e000010-jungfrau.xtc2");
  XtcFileIterator xfi(fd, 0x4000000);

  Dgram* dg = xfi.next();
  const void* bufEnd = ((char*)dg) + xfi.size();
  ConfigIter ci(&(dg->xtc), bufEnd);

  //AreaDetectorJungfrau& det = (AreaDetectorJungfrau&)*getAreaDetector("jungfrau", ci);
  AreaDetector& det = *getAreaDetector("jungfrau", ci);

  //det.AreaDetector::process_config();
  //det.process_config();
  det.print_config();

  unsigned neventreq=5;
  unsigned nevent=0;
  while ((dg = xfi.next())) {
      if (nevent>=neventreq) break;

      printf("evt:%04d %s", nevent, str_dg_info(dg).c_str());

      DataIter di(&(dg->xtc), bufEnd);

      raw_t* d;
      det.raw<raw_t>(di, d, "frame");
      printf("\n == raw data pointer: %d %d %d %d %d\n", d[0],d[1],d[2],d[3],d[4]);

      NDArray<raw_t> nda;
      det.raw<raw_t>(di, nda);
      std::cout << " == raw data nda: " << nda << '\n';

      nevent++;
  }
  ::close(fd);
}

//-------------------

void test_AreaDetectorJungfrau(int argc, char* argv[]) {
  MSG(INFO, "In test_AreaDetectorJungfrau");

  //typedef uint16_t raw_t; // in psalg/calib/AreaDetectorTypes.hh

  int fd = file_descriptor("data-xpptut15-r0430-e000010-jungfrau.xtc2");
  XtcFileIterator xfi(fd, 0x4000000);

  Dgram* dg = xfi.next();
  const void* bufEnd = ((char*)dg) + xfi.size();
  ConfigIter ci(&(dg->xtc), bufEnd);

  AreaDetectorJungfrau det("jungfrau", ci);
  //det.process_config();
  //std::cout << str_config_names(ci) << '\n';
  det.print_config();

  unsigned neventreq=5;
  unsigned nevent=0;
  while ((dg = xfi.next())) {
      if (nevent>=neventreq) break;

      printf("evt:%04d\n", nevent);
      //print_dg_info(dg);

      DataIter di(&(dg->xtc), bufEnd);
      NDArray<raw_jungfrau_t>& nda1 = det.raw(di);
      std::cout << "  == raw data nda for DataIter: " << nda1 << '\n';

      DescData& ddata = di.desc_value(ci.namesLookup());
      NDArray<raw_jungfrau_t>& nda2 = det.raw(ddata);
      std::cout << "  == raw data nda for DescData: " << nda2 << '\n';

      nevent++;
  }
  ::close(fd);
}

//-------------------

void test_AreaDetectorCspad(int argc, char* argv[]) {
  MSG(INFO, "In test_AreaDetectorCspad");

  //typedef uint16_t raw_t; // in psalg/calib/AreaDetectorTypes.hh

  //int fd = file_descriptor("data-cxid9114-r0089-e000010-cspad.xtc2");
  int fd = file_descriptor("data-xpptut15-r0430-e000010-jungfrau.xtc2");
  XtcFileIterator xfi(fd, 0x4000000);

  Dgram* dg = xfi.next();
  const void* bufEnd = ((char*)dg) + xfi.size();
  ConfigIter ci(&(dg->xtc), bufEnd);

  AreaDetectorJungfrau det("jungfrau", ci);
  //AreaDetectorCspad det("jungfrau", ci);
  //det.process_config();
  //std::cout << str_config_names(ci) << '\n';
  det.print_config();

  unsigned neventreq=5;
  unsigned nevent=0;
  while ((dg = xfi.next())) {
      if (nevent>=neventreq) break;
      nevent++;

      printf("evt:%04d\n", nevent);
      //print_dg_info(dg);

      DataIter di(&(dg->xtc), bufEnd);
      NDArray<raw_jungfrau_t>& nda1 = det.raw(di);
      std::cout << "  == raw data nda for DataIter: " << nda1 << '\n';


      DescData& ddata = di.desc_value(ci.namesLookup());
      NDArray<raw_jungfrau_t>& nda2 = det.raw(ddata);
      std::cout << "  == raw data nda for DescData: " << nda2 << '\n';

  }
  ::close(fd);
}

//-------------------

void test_AreaDetectorPnccd(int argc, char* argv[]) {
  MSG(INFO, "In test_AreaDetectorPnccd");

  //typedef uint16_t raw_t; // in psalg/calib/AreaDetectorTypes.hh
  //int fd = file_descriptor("data-sxrx20915-r0206-e000010-pnccd.xtc2");

  int fd = file_descriptor("data-xpptut15-r0450-e000010-pnccd.xtc2");
  XtcFileIterator xfi(fd, 0x4000000);

  Dgram* dg = xfi.next();
  const void* bufEnd = ((char*)dg) + xfi.size();
  ConfigIter ci(&(dg->xtc), bufEnd);

  AreaDetectorPnccd det("pnccd_0001", ci);
  det._class_msg("some string");

  // set parameters to get calib files.
  // In future this info should be supplied directly from xtc file...
  det.set_expname("xpptut15");
  det.set_runnum(450);

  std::cout << "== det.expname     : " << det.expname() << '\n';
  std::cout << "== det.runnum      : " << det.runnum()  << '\n';
  std::cout << "== det.detname     : " << det.detname() << '\n';
  std::cout << "== det.query       : " << det.query()   << '\n';

  det.load_calib_constants();

  det._set_indexes_config(ci);
  det.print_config_indexes();

  std::cout << str_config_names(ci) << '\n';
  det.print_config();

  unsigned neventreq=5;
  unsigned nevent=0;
  bool first_entrance = true;
  while ((dg = xfi.next())) {
      if (nevent>=neventreq) break;

      printf("evt:%04d\n", nevent);

      print_dg_info(dg);

      DataIter di(&(dg->xtc), bufEnd);

      if(first_entrance) {
        first_entrance = false;
        det._set_indexes_data(di);
        det.print_data_indexes();
      }

      det.print_data(di);

      NDArray<pnccd_raw_t>& nda1 = det.raw(di);
      std::cout << "  ==   raw data nda for DataIter: " << nda1 << '\n';

      //DescData& dd = di.desc_value(ci.namesLookup());
      //NDArray<pnccd_raw_t>& nda2 = det.raw(dd);
      //std::cout << "  ==   raw data nda for DescData: " << nda2 << '\n';

      NDArray<pnccd_calib_t>& nda3 = det.calib(di);
      std::cout << "  == calib data nda for DataIter: " << nda3 << '\n';
      nevent++;
  }

  ::close(fd);
}

//-------------------

void test_AreaDetectorOpal(int argc, char* argv[]) {
  MSG(INFO, "In test_AreaDetectorOpal");

  int fd = file_descriptor("data-xppl1316-r0193-e000010-opal1k.xtc2");
  XtcFileIterator xfi(fd, 0x4000000);

  Dgram* dg = xfi.next();
  const void* bufEnd = ((char*)dg) + xfi.size();
  ConfigIter ci(&(dg->xtc), bufEnd);

  AreaDetectorOpal det("opal_0001", ci);
  det._class_msg("XXXX just a test string");

  // set parameters to get calib files.
  // In future this info should be supplied directly from xtc file...
  det.set_expname("xppl1316");
  det.set_runnum(193);

  std::cout << "== det.expname     : " << det.expname() << '\n';
  std::cout << "== det.runnum      : " << det.runnum()  << '\n';
  std::cout << "== det.detname     : " << det.detname() << '\n';
  std::cout << "== det.query       : " << det.query()   << '\n';

  //det.load_calib_constants();

  det.print_config_indexes();

  std::cout << "XXXX str_config_names(ci):\n" << str_config_names(ci) << '\n';
  det.print_config();

  //===========================
  //exit(0);
  //===========================

  unsigned neventreq=5;
  unsigned nevent=0;
  bool first_entrance = true;
  while ((dg = xfi.next())) {
      if (nevent>=neventreq) break;
      nevent++;

      printf("evt:%04d\n", nevent);

      print_dg_info(dg);

      DataIter di(&(dg->xtc), bufEnd);

      if(first_entrance) {
        first_entrance = false;
        det._set_indexes_data(di);
        det.print_data_indexes(di);
      }

      det.print_data(di);

      NDArray<opal_raw_t>& nda1 = det.raw(di);
      std::cout << "  ==   raw data nda for DataIter: " << nda1 << '\n';

      //DescData& dd = di.desc_value(ci.namesLookup());
      //NDArray<opal_raw_t>& nda2 = det.raw(dd);
      //std::cout << "  ==   raw data nda for DescData: " << nda2 << '\n';

      //NDArray<opal_calib_t>& nda3 = det.calib(di);
      //std::cout << "  == calib data nda for DataIter: " << nda3 << '\n';

  }

  ::close(fd);
}


//-------------------

void test_misc(int argc, char* argv[]) {

  MSG(INFO, "In test_misc");
  int fd = file_descriptor("data-xppl1316-r0193-e000010-opal1k.xtc2");
  //XtcFileIterator xfi(fd, 0x4000000);
  std::cout << "file_descriptor: " << fd << '\n';

  static std::size_t size = 100;
  char buf[size];
  ssize_t sz = ::read(fd, &buf[0], size);
  std::cout << "size of read: " << sz << '\n';

  std::cout << "raw content: " << '\n';
  unsigned i=25; char* p=&buf[i];
  for(; i<55; i++, p++)
      std::cout << i << " p:" << (void*) p << " v:" << std::bitset<8>(*p) << '\n';

  ::close(fd);


  std::cout << "sizeof(float)        : " << sizeof(float) << '\n';
  std::cout << "sizeof(double)       : " << sizeof(double) << '\n';
  std::cout << "sizeof(int)          : " << sizeof(int) << '\n';
  std::cout << "sizeof(unsigned)     : " << sizeof(unsigned) << '\n';
  std::cout << "sizeof(char*)        : " << sizeof(char*) << '\n';
  std::cout << "sizeof(std::size_t)  : " << sizeof(std::size_t) << '\n';
  std::cout << "sizeof()             : " << sizeof(int) << '\n';
}

//-------------------
//-------------------
//-------------------
//-------------------

std::string usage(const std::string& tname="")
{
  std::stringstream ss;
  if (tname == "") ss << "Usage command> test_AreaDetector <test-number>\n  where test-number";
  if (tname == "" || tname=="0"	) ss << "\n  0  - test_AreaDetector()         - demo of derived AreaDetectorJungfrau methods";
  if (tname == "" || tname=="1"	) ss << "\n  1  - test_AreaDetector_base()    - demo of generic methods of the base class";
  if (tname == "" || tname=="2"	) ss << "\n  2  - test_AreaDetector_mix()     - demo of mixed output of different objects" ;
  if (tname == "" || tname=="3"	) ss << "\n  3  - test_AreaDetectorJungfrau() - demo of jungfrau specific methods";
  if (tname == "" || tname=="4"	) ss << "\n  4  - test_AreaDetectorPnccd()    - demo of pnccd specific methods";
  if (tname == "" || tname=="5"	) ss << "\n  5  - test_AreaDetectorOpal()     - demo of opal specific methods";
  if (tname == "" || tname=="9" ) ss << "\n  9  - test_getAreaDetector()";
  if (tname == "" || tname=="50") ss << "\n 50  - test_misc()";
  ss << '\n';
  return ss.str();
}

//-------------------

int main(int argc, char **argv) {

  print_hline(80,'_');
  //MSG(INFO, LOGGER.tstampStart() << " Logger started"); // optional record
  //LOGGER.setLogger(LL::DEBUG, "%H:%M:%S.%f");           // set level and time format
  LOGGER.setLogger(LL::INFO, "%H:%M:%S.%f");           // set level and time format

  cout << usage();
  print_hline(80,'_');
  std::string tname((argc>1) ? argv[1] : "0");
  cout << usage(tname);

  if      (tname=="0") test_AreaDetector(argc, argv);
  else if (tname=="1") test_AreaDetector_base(argc, argv);
  else if (tname=="2") test_AreaDetector_mix(argc, argv);
  else if (tname=="3") test_AreaDetectorJungfrau(argc, argv);
  else if (tname=="4") test_AreaDetectorPnccd(argc, argv);
  else if (tname=="5") test_AreaDetectorOpal(argc, argv);
  else if (tname=="50") test_misc(argc, argv);
  else MSG(WARNING, "Undefined test name: " << tname);

  print_hline(80,'_');
  return EXIT_SUCCESS;
}

//-------------------
