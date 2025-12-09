
// 2025-12-01 created by Chris to access data from file in standalone test of calib
// compile Rick's example: /sdf/home/c/claus/git/lcls2/psdaq/drpGpu/calibTest2.cc
// compile this test: g++ -O3 -I . -o test_calib_standalone test_calib_standalone.cc -lpthread

// cd .../lcls2/psana/psana/pycalgos/
// ./test_make_data_for_standalone.py 1 #  make binary file with calib constants
// ./test_make_data_for_standalone.py 2 #  make binary file with raw data
// ./test_make_data_for_standalone.py   #  make BOTH calib constants and raw data files
// ./test_calib_standalone
// mpirun -n 2 ./test_calib_standalone

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <cstddef>  // for size_t
#include <stdint.h> // for uint8_t, uint16_t etc.
//#include <string>
#include <iostream> // std::cout
#include <chrono> // time
#include <sched.h> // sched_getcpu
#include <fstream> // for ifstream

#define time_point_t std::chrono::steady_clock::time_point
#define time_now std::chrono::steady_clock::now
#define duration_us std::chrono::duration_cast<std::chrono::microseconds>

using namespace std;

//#include "UtilsDetector.hh"
//using namespace utilsdetector;

//typedef utilsdetector::time_t time_dt;
typedef double   time_dt;
typedef uint16_t rawd_t;
typedef float    peds_t;
typedef float    gain_t;
typedef float    out_t;
typedef float    cc_t;
typedef uint8_t  mask_t;
typedef uint32_t sizeb_t;

  uint16_t B15 =  040000; // 16384 or 1<<14 (15-th bit starting from 1);
  uint16_t B16 = 0100000; // 32768 or 2<<14 or 1<<15; // 16384 or 1<<14 (16-th bit starting from 1);
  uint16_t BGN = 0140000; // 49152 or 3<<14
  uint16_t MDA = 0x3fff;  // 16383 or (1<<14)-1 - 14-bit mask for data bits
  uint16_t BSH = 14;      // v>>14 bits shift to get gain bits in 0 and 1 bit

void print_sizesof() {
  printf("\n  sizeof(double)  : %d", sizeof(double));
  printf("\n  sizeof(float)   : %d", sizeof(float));
  printf("\n  sizeof(int)     : %d", sizeof(int));
  printf("\n  sizeof(uint32_t): %d", sizeof(uint32_t));
  printf("\n  sizeof(uint16_t): %d", sizeof(uint16_t));
  printf("\n  sizeof(uint8_t) : %d", sizeof(uint8_t));
  printf("\n  sizeof(size_t)  : %d", sizeof(size_t));
  printf("\n  sizeof(sizeb_t) : %d", sizeof(sizeb_t));
  printf("\n  sizeof(ssize_t) : %d", sizeof(ssize_t));
  printf("\n  sizeof(unsigned): %d", sizeof(unsigned));
  printf("\n  sizeof(unsigned int)  : %d", sizeof(unsigned int));
  printf("\n  sizeof(unsigned short): %d", sizeof(unsigned short));
  printf("\n  sizeof(unsigned long) : %d", sizeof(unsigned long));
  printf("\n  sizeof(rawd_t)  : %d", sizeof(rawd_t));
  printf("\n  sizeof(cc_t)    : %d", sizeof(cc_t));
  printf("\n  sizeof(time_dt) : %d", sizeof(time_dt));
  printf("\n\n");
}


std::string login_name() {
    #include <unistd.h> // For getlogin()
    char* name = getlogin();
    if (name != nullptr) {
        std::cout << "Current login name: " << name << std::endl;
    } else {
        std::cerr << "Could not retrieve login name." << std::endl;
    }
    return std::string(name);
}


time_t calib_jungfrau_v3(const rawd_t *raw, const cc_t *cc, const sizeb_t& size, const sizeb_t& size_blk, out_t *out)
{
  // V3 - assuming that
  // * constants are defined as cc[4][<number-of-pixels>][2] - Rick's shape,
  //   where 4 stands for combinations of gain bits, 00,01,10,11, and 2 for peds-offset then gain*mask.
  //   cc.shape = (<4-gain-ranges>, <number-of-pixels-in detector>, <2-for-peds-and-gains>) = (4, npix, 2)
  // * raw and out have letgth of size, where
  // size_blk IS NOT USED

  time_point_t t0 = time_now();
  sizeb_t icc;
  rawd_t rawt;
  for (sizeb_t i=0; i<size; ++i) {
    rawt = raw[i];
    icc = 2*(i + size*(rawt >> BSH)); // index of calibration constants of V3 (rawt >> BSH) & 0x3)
    //std::cout << "  peds-offset:" << cc[icc] << " gain:" << cc[icc+1] << std::endl;
    out[i] = ((rawt & MDA) - cc[icc]) * cc[icc+1];
  }
  return duration_us(time_now() - t0).count();
}

void check_file_is_available(const std::string& filename) {
  std::ifstream f(filename);
  if(f.good()) {
    cout << "use file: " << filename << endl;
    return;
  }
  cout << "NOT AVAILABLE FILE: " << filename << endl;
  cout << "Before running this test try command:\n  ./test_make_data_for_standalone.py\nto make calib constants and data in tmp files" << endl;
  exit(EXIT_FAILURE);
}

void test_calib() {

  std::string name = login_name();
  const std::string& dir_tmp("/lscratch/" + name + "/tmp/");
  const std::string& fname_cc(dir_tmp + "calibcons_v3.dat");
  const std::string& fname_data(dir_tmp + "raw_data_mfx100848724_r051_e000100.dat");
  //const std::string& fname_data("/sdf/data/lcls/ds/xpp/xpptut15/scratch/cpo/cpojunk.dat");

  check_file_is_available(fname_cc);
  check_file_is_available(fname_data);
  //return;

  int icpu = sched_getcpu();
  std::stringstream sscpu; sscpu << "cpu-" << std::setfill('0') << std::setw(3) << std::right << icpu;
  std::string scpu = sscpu.str();

  cout << scpu << endl;

  unsigned events = 100;

  sizeb_t npix = 32*512*1024;
  sizeb_t size_blk = npix;
  sizeb_t sizecc = 4*npix*2;
  ssize_t lena;
  time_dt dt_us;

  print_sizesof();

  cc_t* ccons = (cc_t*)malloc(sizecc*sizeof(cc_t));
  FILE *fcc = fopen(fname_cc.c_str(),"rb");
  lena = fread(ccons, sizeof(float), sizecc, fcc);
  printf("expected size calib cons (4*npix*2): %d\n", sizecc);
  printf("length: %d ccons[0:3]: %f %f %f %f\n", lena, ccons[0], ccons[1], ccons[2], ccons[3]);

  out_t* out = (out_t*)malloc(npix*sizeof(out_t));
  rawd_t* raw = (rawd_t*)malloc(npix*sizeof(rawd_t));
  FILE *fdat = fopen(fname_data.c_str(),"rb");

  time_dt sum_dt = 0;
  unsigned nevt = 0;
  do {
    nevt++;
    lena = fread(raw, sizeof(rawd_t), npix, fdat);
    dt_us = calib_jungfrau_v3(raw, ccons, npix, size_blk, out);
    cout << scpu
         << "  evt: " << setw(3) << right << nevt << "  dt,us: " << fixed << setw(8) << setprecision(0) << dt_us
	 << fixed << setprecision(1)
    	 << "  raw: "; for (int i=0; i<5; i++){cout << setw(6) << right << raw[i] << ' ';}
    cout << "  out: "; for (int i=0; i<5; i++){cout << setw(8) << right << out[i] << ' ';}
    cout << '\n';
    sum_dt += dt_us;
  } while (lena>0 and nevt<events);

  time_dt dt_ave = sum_dt/(double)nevt;
  time_dt fr_Hz = (double)1e6/dt_ave;

  stringstream ssres; ssres << "*** " << scpu << " events:" << nevt << "  <dt>,us: " << dt_ave << "  f,Hz: " << fr_Hz <<'\n';
  string sres = ssres.str();
  cout << sres;
  FILE *file = fopen("summary.txt","a");
  size_t reclen = fwrite(sres.c_str(), sizeof(char), sres.length(), file);
  fclose(file);
  cout << "see results in summary.txt" << endl;
}


void test_event_loop_cpo() {
 unsigned size = 32*512*1024;
 uint16_t* data = (uint16_t*)malloc(size*sizeof(uint16_t));
 FILE *f = fopen("/sdf/data/lcls/ds/xpp/xpptut15/scratch/cpo/cpojunk.dat","rb");
 ssize_t lena;
 unsigned nevt = 0;
 do {
   nevt++;
   lena = fread(data,sizeof(uint16_t),size,f);
   printf("nevt: %d length: %d\n",nevt,lena);
 } while (lena>0);
}


int main() {
  //test_event_loop_cpo();
  test_calib();
  return 0;
}
