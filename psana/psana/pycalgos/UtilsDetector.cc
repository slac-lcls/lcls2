
#include "pycalgos/UtilsDetector.hh"
#include <chrono> // time

#define time_point_t std::chrono::steady_clock::time_point
#define time_now std::chrono::steady_clock::now
#define duration_us std::chrono::duration_cast<std::chrono::microseconds>

using namespace std;

namespace utilsdetector {

// out = (raw-peds)*gain if mask>0 else 0
// returns (double) execution time, us

  /*
//time_t calib_std_2(rawd_t *raw, peds_t *peds, gain_t *gain, mask_t *mask, const size_t& size, const rawd_t databits, out_t *out)
time_t calib_std_2(const rawd_t *raw, const peds_t *peds, const gain_t *gain, const mask_t *mask, const size_t& size, const rawd_t databits, out_t *out)
{
  time_point_t t0 = time_now();
  for (size_t i=0; i<size; ++i) {
    //out[i] = mask[i]>0 ? ((raw[i] & databits) - peds[i])*gain[i] : 0;
    out[i] = ((raw[i] & databits) - peds[i])*gain[i]*mask[i];
  }
  return duration_us(time_now() - t0).count();
}


  //time_t calib_std_v2(rawd_t *raw, peds_t *peds, gain_t *gain, mask_t *mask, const size_t& size, const rawd_t databits, out_t *out)
time_t calib_std_v3(const rawd_t *raw, const peds_t *peds, const gain_t *gain, const mask_t *mask, const size_t& size, const rawd_t databits, out_t *out)
{
//RAWD_T* end = raw+size;
  time_point_t t0 = time_now();
  while (raw<raw+size) {
    *out = ((*raw & databits) - *peds)*(*gain)*(*mask);
     raw++; peds++; gain++; mask++; out++;
  }
  return duration_us(time_now() - t0).count();
}
  */

time_t calib_std(const rawd_t *raw, const peds_t *peds, const gain_t *gain, const mask_t *mask, const size_t& size, const rawd_t databits, out_t *out)
{
  const rawd_t *r = raw;
  const peds_t *p = peds;
  const gain_t *g = gain;
  const mask_t *m = mask;
  out_t *o = out;
  time_point_t t0 = time_now();
  while (r<raw+size) {
    *o++ = ((*r++ & databits) - *p++)*(*g++)*(*m++);
  }
  return duration_us(time_now() - t0).count();
}

}; // namespace utilsdetector

// EOF

