
// g++ -O3 -o test_calib_sim test_calib_sim.cc
//
// srun --partition milano --account lcls:prjdat21 -n 128 --time=05:00:00 --exclusive --pty /bin/bash
//
// ../lcls2/psana/psana/pycalgos/test_calib_sim 2 a01
// mpirun -n  4 ../lcls2/psana/psana/pycalgos/test_calib_sim 2 v04
// mpirun -n  8 ../lcls2/psana/psana/pycalgos/test_calib_sim 2 v08
// mpirun -n 16 ../lcls2/psana/psana/pycalgos/test_calib_sim 2 v16
// mpirun -n 32 ../lcls2/psana/psana/pycalgos/test_calib_sim 2 v32
// mpirun -n 64 ../lcls2/psana/psana/pycalgos/test_calib_sim 2 v64
// mpirun -n 80 ../lcls2/psana/psana/pycalgos/test_calib_sim 2 v80
// mpirun -n 96 ../lcls2/psana/psana/pycalgos/test_calib_sim 2 v96
// mpirun -n 80 --bynode ../lcls2/psana/psana/pycalgos/test_calib_sim 2 v80

#include <string>
#include <sstream> // std::stringstream
#include <random>
#include <chrono>  // time
#include <iomanip>
#include <iostream>
#include <fstream>
#include <unistd.h> // usleep
#include <time.h>
#include <sched.h> // sched_getcpu


#define time_t std::chrono::steady_clock::time_point
#define time_now std::chrono::steady_clock::now
#define duration_us std::chrono::duration_cast<std::chrono::microseconds> //nanoseconds


double random_value(){
  // returns random value in the range [0,1)
  std::srand(std::time(NULL)); // reset seed
  return (double)rand() / (RAND_MAX + 1.0);
}

void random_array_v0(const size_t& size, double *out){
  // returns array of random values in the range [0,1)
  std::srand(std::time(NULL)); // reset seed
  for (double *p = out; p < out + size; p++) {
    *p = (double)rand() / (RAND_MAX + 1.0);
  }
}

void random_array(const double& rand_min, const double& rand_max, const size_t& size, double *out){
  // returns array of random values in the range [rand_min, rand_max]
  std::random_device rd;
  std::mt19937 gen(rd()); // reset seed
  std::uniform_real_distribution<> dis(rand_min, rand_max); //uniform distribution in range
  for (double *p = out; p < out + size; p++) {
    *p = dis(gen);
  }
}

template <typename T>
void random_array_0or1(const double& p1, const size_t& size, T *out){
  double* w = (double*)malloc(size*sizeof(double));
  random_array(0., 1., size, w);
  for (int i=0; i<size; i++){out[i] = (T)((w[i]<p1) ? 1 : 0);}
  delete w;
}

template <typename T>
void standard_normal_value_v0(const double& mean, const double& stddev, T* out){
  // returns standard normal value
  std::default_random_engine engine;
  engine.seed(time_now().time_since_epoch().count());
  std::normal_distribution<double> distribution(mean, stddev);
  *out = distribution(engine);
}

template <typename T>
void standard_normal_value(const double& mean, const double& stddev, T* out)
{
  // returns standard normal value
    std::random_device rd;
    std::mt19937 generator(rd()); // time dependent seed
    std::normal_distribution<double> distribution(mean, stddev);
    *out = (T)distribution(generator);
}

template <typename T>
void standard_normal_array(const double& mean, const double& stddev, const size_t& size, T *out){
  std::random_device rd;
  std::mt19937 generator(rd()); // time dependent seed
  std::normal_distribution<double> distribution(mean, stddev);
  for (T *p = out; p < out + size; p++) {
    *p = (T)distribution(generator);
  }
}

void test_standard_normal_value()
{
  #define GTYPE int // float // double
  GTYPE v;
  double mean=100;
  double sigma=5;
  for (int i=0; i<10; ++i) {
    standard_normal_value<GTYPE>(mean, sigma, &v);
    //standard_normal_value_v0<GTYPE>(mean, sigma, &v);
    std::cout << "i: " << i << " rnd: " << v << std::endl;
  }
}

void test_random_array()
{
  #define SIZER 10000 // 2162688 //16*352*384
  double out[SIZER];
  time_t t0 = time_now();
  //random_array(SIZER, &out[0]);
  //random_array(0., 1., SIZER, &out[0]);
  random_array_0or1<double>(0.6, SIZER, &out[0]);
  time_t tf = time_now();

  for (int i=0; i<SIZER; ++i) {
    if(i%1000 == 0) {
      std::cout << "i: " << i << " rnd: " << out[i] << std::endl;
    }
  }
  std::cout << "Consumed time: " << duration_us(tf - t0).count() << " us" << std::endl;
}

void test_standard_normal_array()
{
  #define VTYPE float //int // float // double
  #define SIZE 1000000 // 2162688 //16*352*384
  #define TOFSET 1729870000 // 1729875356.363
  VTYPE out[SIZE];
  double mean=100;
  double sigma=5;

  time_t t0 = time_now();
  standard_normal_array<VTYPE>(mean, sigma, SIZE, &out[0]);
  time_t tf = time_now();

  for (int i=0; i<SIZE; ++i) {
    if(i%1000 == 0) {
      std::cout << "i: " << i << " rnd: " << out[i] << std::endl;
    }
  }
  std::cout << "Consumed time: " << duration_us(tf - t0).count() << " us" << std::endl;
}

void test_sizeof(){
    std::cout << "sizeof(std::size_t): "  << sizeof(std::size_t) << std::endl;
    std::cout << "sizeof(int): "  << sizeof(int) << std::endl;
    std::cout << "sizeof(uint): "  << sizeof(uint) << std::endl;
    std::cout << "sizeof(float): "  << sizeof(float) << std::endl;
    std::cout << "sizeof(double): "  << sizeof(double) << std::endl;
    std::cout << "sizeof(char): "  << sizeof(char) << std::endl;
    std::cout << "sizeof(16*352*384): "  << sizeof(16*352*384) << std::endl;
}

void test_time(){
  time_t t0 = time_now();
  test_standard_normal_array();
  time_t tf = time_now();
  std::cout << "Consumed time: " << duration_us(tf - t0).count() << " us" << std::endl;
}

// ==================
// TEST det.raw.calib

#include <stdint.h>
#include <stdlib.h>
#include <cstdint>  // uint8_t

#define PSIZE 16*352*384 // // 100000 // 2162688 = 16*352*384
#define NLOOPS 100
#define EVENTS 100
#define M14 0x3fff  // 16383 or (1<<14)-1 - 14-bit mask

#define RAWD_T uint16_t
#define MASK_T uint8_t
#define GAIN_T float
#define PEDS_T float
#define REST_T float

struct pixstr {
  MASK_T mask;
  PEDS_T ped;
  GAIN_T gain;
  REST_T rest;
};

void calib(RAWD_T* raw, pixstr* pu) {
  RAWD_T* end = raw+PSIZE;
  while (raw<end) {
    //(*pu).rest = ((*raw & M14) - (*pu).ped) * ((*pu).gain) * ((*pu).mask);
    pu->rest = ((*raw & M14) - (pu->ped)) * (pu->gain) * (pu->mask);
     raw++; pu++;
  }
}

void calib_v0(RAWD_T* raw, MASK_T* mask, GAIN_T* gain, PEDS_T* ped, REST_T* res) {
  RAWD_T* end = raw+PSIZE;
  while (raw<end) {
    *res = ((*raw & M14) - *ped)*(*gain)*(*mask);
     raw++; ped++; gain++; mask++; res++;
  }
}

void test_calib_simulation()
{
  time_t t0 = time_now();

  RAWD_T* rawd = (RAWD_T*)malloc(EVENTS*PSIZE*sizeof(RAWD_T));
  MASK_T* mask = (MASK_T*)malloc(PSIZE*sizeof(MASK_T));
  PEDS_T* peds = (PEDS_T*)malloc(PSIZE*sizeof(PEDS_T));
  GAIN_T* gain = (GAIN_T*)malloc(PSIZE*sizeof(GAIN_T));
  REST_T* rest = (REST_T*)malloc(PSIZE*sizeof(REST_T));

  std::cout << "test_calib_simulation: time for malloc: " << duration_us(time_now() - t0).count() << " us" << std::endl;

  t0 = time_now();
  standard_normal_array<RAWD_T>(1000., 10., PSIZE*EVENTS, rawd);
  standard_normal_array<PEDS_T>(1000., 10., PSIZE, peds);
  standard_normal_array<GAIN_T>(20., 1., PSIZE, gain);
  random_array_0or1<MASK_T>(0.9, PSIZE, mask);

  std::cout << "time for random data and constants: " << duration_us(time_now() - t0).count() << " us" << std::endl;

  std::cout << "\nrawd: "; for (int i=0; i<10; i++){std::cout << rawd[i] << " ";}
  std::cout << "\npeds: "; for (int i=0; i<10; i++){std::cout << peds[i] << " ";}
  std::cout << "\ngain: "; for (int i=0; i<10; i++){std::cout << gain[i] << " ";}
  std::cout << "\nmask: "; for (int i=0; i<10; i++){std::cout << unsigned(mask[i]) << " ";}
  std::cout << std::endl;

  std::cout << "events: " << std::to_string(EVENTS) << " panel size:" << std::to_string(PSIZE) << std::endl;

  t0 = time_now();
  for (int i=0; i<EVENTS; i++){
     calib_v0(rawd+i*PSIZE, mask, gain, peds, rest);
  }
  std::cout << " time per event: " << duration_us(time_now() - t0).count()/EVENTS << " us" << std::endl;
}

double time_sec(struct timespec& t){return t.tv_sec + t.tv_nsec * 1e-9;}

//#include <mpi.h>
//int rank;
//MPI_Init(&argc, &argv);
//MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//MPI_Finalize();

void test_calib_simulation_mpi(int argc, char* argv[])
{
  int icpu = sched_getcpu();
  std::stringstream sscpu; sscpu << "cpu-" << std::setfill('0') << std::setw(3) << std::right << icpu;
  std::string scpu = sscpu.str();

  time_t t0 = time_now();

  RAWD_T* rawd = (RAWD_T*)malloc(EVENTS*PSIZE*sizeof(RAWD_T));
  MASK_T* mask = (MASK_T*)malloc(PSIZE*sizeof(MASK_T));
  REST_T* rest = (REST_T*)malloc(PSIZE*sizeof(REST_T));
  PEDS_T* peds = (PEDS_T*)malloc(PSIZE*sizeof(PEDS_T));
  GAIN_T* gain = (GAIN_T*)malloc(PSIZE*sizeof(GAIN_T));
  pixstr* pixs = (pixstr*)malloc(PSIZE*sizeof(pixstr));

  std::cout << scpu
	    << " test_calib_simulation_mpi time for malloc: "
	    << duration_us(time_now() - t0).count() << " us" << std::endl;

  t0 = time_now();
  standard_normal_array<RAWD_T>(1000., 10., PSIZE*EVENTS, rawd);
  standard_normal_array<PEDS_T>(1000., 10., PSIZE, peds);
  standard_normal_array<GAIN_T>(20., 1., PSIZE, gain);
  random_array_0or1<MASK_T>(0.9, PSIZE, mask);

  for (int i=0; i<PSIZE; i++){
    pixs[i].mask = mask[i];
    pixs[i].ped  = peds[i];
    pixs[i].gain = gain[i];
    pixs[i].rest = rest[i];
  }

  if (icpu == 0){
    std::cout << scpu << " time for random data and constants: "
	      << duration_us(time_now() - t0).count() << " us";
    std::cout << "\n  rawd: "; for (int i=0; i<10; i++){std::cout << rawd[i] << ' ';}
    std::cout << "\n  peds: "; for (int i=0; i<10; i++){std::cout << peds[i] << ' ';}
    std::cout << "\n  gain: "; for (int i=0; i<10; i++){std::cout << gain[i] << ' ';}
    std::cout << "\n  mask: "; for (int i=0; i<10; i++){std::cout << unsigned(mask[i]) << ' ';}
    std::cout << "\n  " << scpu << " events: " << std::to_string(EVENTS)
	      << " panel size:" << std::to_string(PSIZE) << std::endl;
  }

  double times_s[NLOOPS];
  double durats_us[NLOOPS];

  struct timespec tbeg, tcur;
  int status = clock_gettime(CLOCK_REALTIME, &tbeg);
  time_t tt0 = time_now();

  for (int n=0; n<NLOOPS; n++){
    status = clock_gettime(CLOCK_REALTIME, &tcur);
    t0 = time_now();

    for (int i=0; i<EVENTS; i++){
      //calib_v0(rawd+i*PSIZE, mask, gain, peds, rest);
      calib(rawd+i*PSIZE, pixs);
    }

    durats_us[n] = duration_us(time_now() - t0).count() / EVENTS;
    times_s[n] = time_sec(tcur) - TOFSET;
  }

  int time_per_event_us = duration_us(time_now() - tt0).count() / EVENTS / NLOOPS;

  std::cout << scpu << " NLOOPS: " << NLOOPS << " EVENTS: " << EVENTS << std::endl;
  std::cout << scpu << " time per event: " << time_per_event_us << " us" << std::endl;

  //std::stringstream fname; fname << "results-" << scpu << "-v80.txt";
  std::string version = (argc>2)? argv[2] : "vXX";
  std::stringstream fname; fname << "results-" << scpu << '-' << version << ".txt";
  std::cout << "save file: " <<  fname.str() << std::endl;

  std::ofstream ofile;
  ofile.open(fname.str());

  for (int i=0; i<NLOOPS; i++){
     ofile << std::setw(3) << std::right << i;
     ofile << std::fixed
               << std::setprecision(6);
     ofile << " t,s:" << std::setw(14) << times_s[i];
     ofile << " dt,us: " << std::setprecision(0) << std::setw(10) << durats_us[i] << std::endl;
  }
  ofile << '\n' << scpu << " time per event: " << time_per_event_us << " us" << std::endl;
  ofile << "begin event loop time_since_epoch, sec: "
	<< std::setw(14) << std::setprecision(3)
	<< time_sec(tbeg)
        << " offset: " << TOFSET << std::endl;
  ofile.close();
}

void test_time_units(){
  struct timespec tbeg, tend;
  int status = clock_gettime(CLOCK_REALTIME, &tbeg);
  time_t t0 = time_now();

  long int tsec = 5;
  std::cout << "sleep for " << tsec << " seconds" << std::endl;
  usleep(tsec*(int)1e6);

  status = clock_gettime(CLOCK_REALTIME, &tend);
  std::cout << " test_time_units (1): "
	    << duration_us(time_now() - t0).count() << " us" << std::endl;

  std::cout << " test_time_units (2): " << time_sec(tend) - time_sec(tbeg) << " sec" << std::endl;
}



int main(int argc, char* argv[])
{
  std::cout << "argc:" << argc<< " argv[0]:" << argv[0] << std::endl;
  std::string tname = (argc>1)? argv[1] : "";

  if      (tname == "")   test_calib_simulation();
  else if (tname == "1")  test_calib_simulation();
  else if (tname == "2")  test_calib_simulation_mpi(argc, &argv[0]);
  else if (tname == "10") test_sizeof();
  else if (tname == "11") test_random_array();
  else if (tname == "12") test_time();
  else if (tname == "13") test_standard_normal_value();
  else if (tname == "14") test_standard_normal_array();
  else if (tname == "15") test_time_units();
  else std::cout << "NOT FOUND TEST " << tname << std::endl;

}

//EOF
