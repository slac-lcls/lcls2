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




#include "xtcdata/xtc/XtcFileIterator.hh"
//#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/ShapesData.hh"
//#include "xtcdata/xtc/DescData.hh"
//#include "xtcdata/xtc/NamesIter.hh"
#include "xtcdata/xtc/ConfigIter.hh"
#include "xtcdata/xtc/DataIter.hh"

#include "xtcdata/xtc/NamesLookup.hh"


#include "psalg/detector/AreaDetectorStore.hh"
//#include "psalg/detector/AreaDetector.hh"

//using namespace std;
//using namespace psalg;
using namespace calib;
using namespace detector;

using namespace std; 
using namespace XtcData;

//-------------------

void print_hline(const uint nchars, const char c) {printf("%s\n", std::string(nchars,c).c_str());}

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

int file_descriptor(int argc, char* argv[]) {

    const char* fname = "/reg/neh/home/cpo/git/lcls2/psana/psana/dgramPort/jungfrau.xtc2";
    std::cout << "xtc file name: " << fname << '\n';

    int fd = open(fname, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Unable to open file '%s'\n", fname);
        exit(2);
    }
    
    return fd;
}

//-------------------

void test_AreaDetector(int argc, char* argv[]) {
  MSG(INFO, "In test_AreaDetector");

  int fd = file_descriptor(argc, argv);
  XtcFileIterator xfi(fd, 0x4000000);

  // get data out of the 1-st datagram configure transition
  Dgram* dg = xfi.next();
  ConfigIter configo(&(dg->xtc));

  NamesLookup& names_map = configo.namesLookup();
  NamesId& namesId = configo.value().namesId();

  //DescData desc_data(configo.value(), names_map[namesId]);
  //DescData& desc_shape = configo.desc_shape();

  NameIndex& nameindex = names_map[namesId];
  Names& names = nameindex.names();

  printf("transition: %d  0/1 = config/data\n", namesId.namesId());

  printf("Names:: detName: %s  detType: %s  detId: %s  segment: %d alg.name: %s\n",
          names.detName(), names.detType(), names.detId(), names.segment(), names.alg().name());

  //=======
  /*

  AreaDetector* det = new AreaDetector("jungfrau");
  std::cout << "detname: " << det->detname() << '\n';

  //query_t query = 123;
  //const NDArray<pedestals_t>& peds = det->pedestals(query);
  //const NDArray<common_mode_t>& cmod = det->common_mode(query);
  //const NDArray<raw_t>& raw = det->raw(query);
  //std::cout << "\n  peds   : " << peds;
  //std::cout << "\n  cmod   : " << cmod;
  //std::cout << "\n  raw    : " << raw;
  //std::cout << '\n';



  //NamesLookup& names_map = configo.namesLookup();


  unsigned neventreq=2;
  unsigned nevent=0;
  while ((dg = xfi.next())) {
      if (nevent>=neventreq) break;
      nevent++;

      DataIter datao(&(dg->xtc));
 
      printf("evt:%04d ==== transition: %s of type: %d time %d.%09d, pulseId %lux, env %ux, "
             "payloadSize %d extent %d\n", nevent,
             TransitionId::name(dg->seq.service()), dg->seq.type(), dg->seq.stamp().seconds(),
             dg->seq.stamp().nanoseconds(), dg->seq.pulseId().value(),
             dg->env, dg->xtc.sizeofPayload(), dg->xtc.extent);

      //DESC_VALUE(desc_data, datao, names_map);
      DescData& desc_data = datao.desc_value(names_map);
      dump("Data values", desc_data);
  }

  //return 0;



  delete det;
  */
  ::close(fd);

}

//-------------------

/*

int test_all(int argc, char* argv[]) {

    int fd = file_descriptor(argc, argv);
    XtcFileIterator xfi(fd, 0x4000000);

    // get data out of the 1-st datagram configure transition
    Dgram* dg = xfi.next();

    ConfigIter configo(&(dg->xtc));
    NamesLookup& names_map = configo.namesLookup();

    //DESC_SHAPE(desc_shape, configo, names_map);
    DescData& desc_shape = configo.desc_shape();
    dump("Config shape", desc_shape);

    //DESC_VALUE(desc_value, configo, names_map);
    DescData& desc_value = configo.desc_value();
    dump("Config values", desc_value);

    unsigned neventreq=2;
    unsigned nevent=0;
    while ((dg = xfi.next())) {
        if (nevent>=neventreq) break;
        nevent++;

        DataIter datao(&(dg->xtc));
 
        printf("evt:%04d ==== transition: %s of type: %d time %d.%09d, pulseId %lux, env %ux, "
               "payloadSize %d extent %d\n", nevent,
               TransitionId::name(dg->seq.service()), dg->seq.type(), dg->seq.stamp().seconds(),
               dg->seq.stamp().nanoseconds(), dg->seq.pulseId().value(),
               dg->env, dg->xtc.sizeofPayload(), dg->xtc.extent);

        //DESC_VALUE(desc_data, datao, names_map);
        DescData& desc_data = datao.desc_value(names_map);
        dump("Data values", desc_data);
    }

    ::close(fd);
    return 0;
}

*/

//-------------------


//-------------------

std::string usage(const std::string& tname="")
{
  std::stringstream ss;
  if (tname == "") ss << "Usage command> test_AreaDetector <test-number>\n  where test-number";
  if (tname == "" || tname=="0"	) ss << "\n  0  - test_AreaDetector()";
  if (tname == "" || tname=="1"	) ss << "\n  1  - test_getAreaDetector()";
  ss << '\n';
  return ss.str();
}

//-------------------

int main(int argc, char **argv) {

  print_hline(80,'_');
  //MSG(INFO, LOGGER.tstampStart() << " Logger started"); // optional record
  LOGGER.setLogger(LL::DEBUG, "%H:%M:%S.%f");           // set level and time format

  cout << usage(); 
  print_hline(80,'_');
  if (argc==1) {return 0;}
  std::string tname(argv[1]);
  cout << usage(tname); 

  if      (tname=="0") test_AreaDetector(argc, argv);
  else if (tname=="1") test_getAreaDetector();
  else MSG(WARNING, "Undefined test name: " << tname);

  print_hline(80,'_');
  return EXIT_SUCCESS;
}

//-------------------
