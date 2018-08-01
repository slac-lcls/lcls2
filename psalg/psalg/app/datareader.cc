// == Build locally
// cd .../lcls2/psalg/build
// make
// == Then run
// psalg/6datareader
// 
// g++ datareader.cc -o mytest datareader

// Build in lcls2/install/bin/
// see .../psalg/psalg/CMakeLists.txt
// cd /reg/neh/home/dubrovin/LCLS/con-lcls2/lcls2
// ./build_all.sh

// Then run it (from lcls2/install/bin/datareader) as 
// datareader

//#include <stdlib.h>
//#include <strings.h>

#include <iostream> // cout, puts etc.
#include <getopt.h>
#include <stdio.h>  // printf
#include <fstream>  // ifstream
#include <iomanip>  // std::setw

#include <dirent.h> // opendir, readdir, closedir, dirent, DIR
#include "psalg/utils/Utils.hh" // files_in_dir, dir_content
//#include "psalg/utils/Logger.hh" // MSG
#include "xtcdata/xtc/DescData.hh" // Array
#include "psalg/calib/ArrayIO.hh" // ArrayIO
#include "psalg/calib/NDArray.hh" // NDArray
#include "psalg/utils/DirFileIterator.hh"
#include "psalg/detector/AreaDetectorStore.hh" // getAreaDetector 
#include "psalg/detector/DetectorStore.hh" // getDetector 
//#include "psalg/detector/AreaDetectorTypes.hh"
//#include "psalg/detector/DetectorTypes.hh"
//#include "psalg/detector/Detector.hh"
#include "psalg/detector/DataSourceSimulator.hh"

using namespace std;
using namespace psalg;
using namespace detector;

//-------------------

void usage(char* name) {
  MSG(INFO, "Usage: " << name << " some stuff about arguments and options");
}

//-------------------

void print_hline(const uint nchars, const char c) {
    printf("%s\n", std::string(nchars,c).c_str());
}

//-------------------
//-------------------
//-------------------
//-------------------

void test_DataSourceSimulator() {
  MSG(INFO, "In test_DataSourceSimulator");

  DataSourceSimulator ds("/reg/neh/home/dubrovin/LCLS/con-detector/work",
		         "nda-xpptut15-r0260-XcsEndstation.0_Epix100a.1");

  AreaDetector* det = dynamic_cast<AreaDetector*>(ds.detector("Epix100a"));
  std::cout << "  Detector from DataSourceSimulator: " << *det << '\n';
      
  for(unsigned i=0;i<5;i++) {
    EventSimulator& evt = ds.events().next();
    std::cout << i << " ndarray: " << evt.nda() << '\n';
  }
}

//-------------------

void test_get_detector() {
  MSG(INFO, "In test_get_detector");
  AreaDetector* det1 = getAreaDetector("Epix100a");
  Detector*     det2 = (Detector*)getDetector("Epix100a");
  AreaDetector* det3 = dynamic_cast<AreaDetector*>(getDetector("Epix100a"));
  std::cout << "AreaDetector obj1: " << *det1 << '\n';
  std::cout << "Detector     obj2: " << *det2 << '\n';
  std::cout << "AreaDetector obj3: " << *det3 << '\n';
  event_t evt = 123;
  std::cout << "det1 size: " << det1->size(evt) << '\n';
  std::cout << "det3 size: " << det3->size(evt) << '\n';
  det3->pedestals(evt);
  det3->common_mode(evt);
  det3->image(evt);
 }

//-------------------

void test_detector() {
  MSG(INFO, "In test_detector");
  Detector det1("Epix100a.");
  Detector det2("Cspad.");
  Detector det3("XCS-YAG-1"); // DDL_DETECTOR
  //print_map_detname_to_dettype();
  print_map_area_detname_to_dettype();
  //print_map_bldinfo_to_dettype();
  std::cout << "Detector object 1 " << det1 << '\n';
  std::cout << "Detector object 2 " << det2 << '\n';
  std::cout << "Detector object 3 " << det3 << '\n';
 }

//-------------------

void test_area_detector() {
  MSG(INFO, "In test_area_detector");
  AreaDetector* det = getAreaDetector("Epix100a.");
  //detector::AreaDetector* det = detector::getAreaDetector("Epix100a.");
  event_t evt = 123;
  //const NDArray<pedestals_t>& peds = modet->pedestals(evt);
  const detector::size_t size = det->size(evt);
  std::cout << "size: " << size << '\n';
  const detector::size_t ndim = det->ndim(evt);
  std::cout << "ndim: " << ndim << '\n';
  const detector::shape_t* shape = det->shape(evt);
  std::cout << "shape[0]: " << shape[0] << '\n';

  NDArray<raw_t> raw = det->raw(evt);
}

//-------------------

void test_image_loop() {
  MSG(INFO, "In test_image_loop");

  std::vector<std::string> v = 
    files_in_dir("/reg/neh/home/dubrovin/LCLS/con-detector/work",
		 "nda-xpptut15-r0260-XcsEndstation.0_Epix100a.1");

  cout << "  found files: \n";
  std::vector<std::string>::iterator it = v.begin();

  unsigned counter=0;
  for (; it != v.end(), counter<2; ++it, ++counter) {
    cout << "  --==-- " << *it << '\n';
    ArrayIO<float> aio(*it);
    //MSG(INFO, "array status: " << aio.str_status());
    NDArray<float>& arr = aio.ndarray();
    cout << "          ndarray: " << arr << '\n';
  }
}

//-------------------

void test_DirFileIterator() {
  MSG(INFO, "In test_DirFileIterator");
  DirFileIterator dfi("/reg/neh/home/dubrovin/LCLS/con-detector/work",
		      "nda-xpptut15-r0260-XcsEndstation.0_Epix100a.1");
  std::string fname;
  while (!(fname=dfi.next()).empty()) {
    std::cout << "next file: " << fname << '\n';
  }
}

//-------------------

void test_files_in_dir() {
  MSG(INFO, "In test_files_in_dir");

  std::vector<std::string> v = 
    files_in_dir("/reg/neh/home/dubrovin/LCLS/con-detector/work",
		 "nda-xpptut15-r0260-XcsEndstation.0_Epix100a.1");

  cout << "  found files: \n";

  std::vector<std::string>::iterator it = v.begin();
  for (; it != v.end(); ++it) cout << "  -- " << *it << '\n';
}

//-------------------

void test_dircontent() {
  MSG(INFO, "In test_dircontent");
  dir_content();
}

//-------------------

void test_ArrayIO() {
  MSG(INFO, "In test_ArrayIO");
  //Array(void *data, uint32_t *shape, uint32_t rank)
  ArrayIO<float> aio("/reg/neh/home/dubrovin/LCLS/con-detector/work/nda-xpptut15-r0260-XcsEndstation.0_Epix100a.1-e000030.txt");

  MSG(INFO, "array status: " << aio.str_status());

  NDArray<float>& arr = aio.ndarray();
  MSG(INFO, "ndarray: " << arr);
}

//-------------------

void test_Array() {

  MSG(DEBUG, "In test_Array");

  //Array(void *data, uint32_t *shape, uint32_t rank)

  float data[] = {1,2,3,4,5,6,7,8,9,10,11,12};
  uint32_t sh[2] = {3,4};
  uint32_t rank = 2;
  XtcData::Array<float> a(data, sh, rank);

  for(uint32_t r=0; r<sh[0]; r++) {
    std::cout << "\nrow " << r << ':';
    for(uint32_t c=0; c<sh[1]; c++)
      std::cout << " " << std::setw(4) << a(r,c);
  }

  std::cout << '\n';
}

//-------------------

void test_NDArray() {

  MSG(DEBUG, "In test_NDArray");

  //Array(void *data, uint32_t *shape, uint32_t rank)

  float data[] = {1,2,3,4,5,6,7,8,9,10,11,12};
  psalg::types::shape_t sh[2] = {3,4}; // uint32_t
  psalg::types::size_t rank = 2;

  NDArray<float> a(sh, rank, data);

  for(uint32_t r=0; r<sh[0]; r++) {
    std::cout << "\nrow " << r << ':';
    for(uint32_t c=0; c<sh[1]; c++)
      std::cout << " " << std::setw(4) << a(r,c);
  }
  
  std::cout << "\narray ndim: " << a.ndim() << '\n';
  std::cout << "\narray size: " << a.size() << '\n';
  std::cout << "ostream array: " << a << '\n';

  uint32_t sh2x6[2] = {2,6};
  a.reshape(sh2x6, rank);
  std::cout << "reshaped array: " << a << '\n';

  std::cout << '\n';
}

//-------------------

void read_file_lines(const char* fname) {

    MSG(DEBUG, "Line-by-line read file " << fname);

    std::ifstream inf(fname);
 
    std::string str; 
    int counter=0;

    while(getline(inf,str)) {
      counter++; 
      if(counter<20) cout << str << '\n'; // break;
    }

    std::cout << "Number of lines in file: " << counter << '\n';
    MSG(INFO, "Number of lines in file: " << counter);

    inf.close();
}

//-------------------

int test_cli(int argc, char **argv)
{
    MSG(INFO, "\n\nJust a test\n\n");

    char *avalue = NULL;
    char *bvalue = NULL;
    char *cvalue = NULL;
    int index;
    int c;
    extern char *optarg;
    extern int optind, optopt; //opterr;
 
    while((c = getopt (argc, argv, ":a:b:c:h")) != -1)
        switch(c)
        {
          case 'a':
            avalue = optarg;
            printf("a: avalue = %s\n",avalue);
            break;
          case 'b':
            bvalue = optarg;
            printf("b: bvalue = %s\n",bvalue);
            break;
          case 'c':
            cvalue = optarg;
            printf("c: cvalue = %s\n",cvalue);
            break;
          case 'h':
            printf("h: ");
            usage(argv[0]);
            break;
          case ':':
            printf("(:) Option -%c requires an argument.\n", optopt);
            usage(argv[0]);
            return EXIT_FAILURE;
          case '?':
            printf("?: Option -%c requires an argument.\n", optopt);
            usage(argv[0]);
            return EXIT_FAILURE;
          default:
            printf("default: You should not get here... option -%c .\n", optopt);
            abort();
        }
 
    printf("End of options: avalue = %s, bvalue = %s, cvalue = %s\n",
            avalue, bvalue, cvalue);
 
    for (index = optind; index < argc; index++)
        printf("Non-option argument %s\n", argv[index]);
    return EXIT_SUCCESS;
}

//-------------------

int main(int argc, char **argv) {

  MSG(INFO, LOGGER.tstampStart() << " Logger started"); // optional record
  //LOGGER.setLogger(LL::DEBUG, "%H:%M:%S.%f");           // set level and time format
  LOGGER.setLogger(LL::INFO, "%H:%M:%S.%f");           // set level and time format

    // test_cli(argc, argv); 
    // print_hline(80,'_');
    // read_file_lines("/reg/neh/home/dubrovin/LCLS/con-detector/work/nda-xpptut15-r0260-XcsEndstation.0_Epix100a.1-e000030.txt");
    print_hline(80,'_');
    // test_Array();         print_hline(80,'_');
    test_ArrayIO();         print_hline(80,'_');
    test_NDArray();         print_hline(80,'_');
    //test_logger_single(); print_hline(80,'_');

    test_dircontent();      print_hline(80,'_');
    test_DirFileIterator(); print_hline(80,'_');

    test_files_in_dir();    print_hline(80,'_');
    test_image_loop();      print_hline(80,'_');

    test_area_detector();   print_hline(80,'_');
    test_detector();        print_hline(80,'_');
    test_get_detector();    print_hline(80,'_');

    test_DataSourceSimulator(); print_hline(80,'_');

    return EXIT_SUCCESS;
}

//-------------------
