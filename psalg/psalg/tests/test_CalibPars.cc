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
#include <stdio.h>  // printf
//#include <getopt.h>
//#include <fstream>  // ifstream
//#include <iomanip>  // std::setw

//#include <dirent.h> // opendir, readdir, closedir, dirent, DIR
//#include "psalg/utils/Utils.hh" // files_in_dir, dir_content
//#include "psalg/utils/Logger.hh" // MSG
//#include "xtcdata/xtc/DescData.hh" // Array
//#include "psalg/calib/ArrayIO.hh" // ArrayIO
//#include "psalg/calib/NDArray.hh" // NDArray
//#include "psalg/utils/DirFileIterator.hh"
//#include "psalg/detector/AreaDetectorStore.hh" // getAreaDetector 
//#include "psalg/detector/DetectorStore.hh" // getDetector 
//#include "psalg/detector/AreaDetectorTypes.hh"
//#include "psalg/detector/DetectorTypes.hh"
//#include "psalg/detector/Detector.hh"
//#include "psalg/detector/DataSourceSimulator.hh"

#include "psalg/calib/CalibPars.hh"

using namespace std;
using namespace psalg;
using namespace calib;
//using namespace detector;

//-------------------

//void usage(char* name) {MSG(INFO, "Usage: " << name << " some stuff about arguments and options");}

//-------------------

void print_hline(const uint nchars, const char c) {printf("%s\n", std::string(nchars,c).c_str());}

//-------------------
//-------------------
//-------------------
//-------------------

void test_CalibPars() {
  MSG(INFO, "In test_CalibPars");
  CalibPars* cp1 = new CalibPars("Epix100a");
  request_t evt = 123;

  std::cout << "cp1.detname(): " << cp1->detname() << '\n';
  cp1->pedestals(evt);
  cp1->common_mode(evt);
 }

//-------------------

int main(int argc, char **argv) {

  MSG(INFO, LOGGER.tstampStart() << " Logger started"); // optional record
  LOGGER.setLogger(LL::INFO, "%H:%M:%S.%f");           // set level and time format

  print_hline(80,'_');
  test_CalibPars(); print_hline(80,'_');

  return EXIT_SUCCESS;
}

//-------------------
