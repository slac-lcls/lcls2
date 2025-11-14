
#include "pycalgos/UtilsDetector.hh"
#include <chrono> // time

#define time_point_t std::chrono::steady_clock::time_point
#define time_now std::chrono::steady_clock::now
#define duration_us std::chrono::duration_cast<std::chrono::microseconds>

using namespace std;

namespace utilsdetector {

  uint16_t B15 =  040000; // 16384 or 1<<14 (15-th bit starting from 1);
  uint16_t B16 = 0100000; // 32768 or 2<<14 or 1<<15; // 16384 or 1<<14 (16-th bit starting from 1);
  uint16_t BGN = 0140000; // 49152 or 3<<14
  uint16_t MDA = 0x3fff;  // 16383 or (1<<14)-1 - 14-bit mask for data bits
  uint16_t BSH = 14;      // v>>14 bits shift to get gain bits in 0 and 1 bit

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


time_t calib_jungfrau_v0(const rawd_t *raw, const peds_t *peds, const gain_t *gain, const mask_t *mask, const size_t& size, out_t *out)
{
  //raw[i], out[i], mask[i] shape:   (<number-of-segs>, 512, 1024)
  //peds[icc], gain[icc]    shape:(3, <number-of-segs>, 512, 1024)

  rawd_t rawi;  // copy of raw[i]
  uint8_t igr; // index of the gain range: 0,1,2
  uint16_t icc; // pixel index in the array of calibration constants

  time_point_t t0 = time_now();
  for (size_t i=0; i<size; ++i) {
    rawi = raw[i];
    igr = rawi >> BSH; if (igr > 1) igr=2; // gain bits gr0/1/2 = 00/01/11. Combination 10 - bad pixel accounted in pixel_status
    icc = igr * size + i;
    out[i] = ((rawi & MDA) - peds[icc])*gain[icc]*mask[i];
  }
  return duration_us(time_now() - t0).count();
}


void calib_jungrfau_blk_v1(const rawd_t *raw, const cc_t *cc, const size_t& size_blk, out_t *out)
{
  //float cc[size_block][8], where 8 is 4-peds and 4 gains per pixel
  //uint8_t igr; // index of the gain range index gr0/1/2 = 00/01/11, combination 10 - bad pixel status
  size_t icc;
  for (size_t i=0; i<size_blk; ++i) {
    icc = 8*i + (raw[i] >> BSH); // index of calibration constants 4*granges for peds and gains
    out[i] = ((raw[i] & MDA) - cc[icc]) * cc[icc+4];
  }
}

  //void calib_jungrfau_blk_v2(const rawd_t *raw, const ccstruct *cc, const size_t& size_blk, out_t *out)
  //{
  //  ccstruct& ccgrpix;
  //  for (size_t i=0; i<size_blk; ++i) {
  //    ccgrpix = cc[i][rawi >> BSH]; // gain range index gr0/1/2 = 00/01/11, combination 10 - bad pixel status
  //    out[i] = (raw[i] & MDA - ccgrpix.pedestal) * ccgrpix.gain;
  //  }
  //}

time_t calib_jungfrau_v1(const rawd_t *raw, const cc_t *cc, const size_t& size, const size_t& size_blk, out_t *out)
{
  // assuming that
  // * constants are defined as cc[<number-of-pixels>][8],
  //   where 8 stands for for ALL 4 combinations of gain bits for peds then gain.
  //   cc.shape = (<number-of-pixels-in detector>, <2-for-peds-and-gains>, <4-gain-ranges>) = (npix, 2, 4)
  // * raw and out have letgth of size, where
  //   size = size_blk * (int)number_of_blocks

  time_point_t t0 = time_now();
  size_t ipx0; // the 0-th pixel index of the block in data and other arrays

  for (size_t ib=0; ib<size/size_blk; ++ib) { // loop over blocks in entire data array of size
    ipx0 = ib*size_blk;
    calib_jungrfau_blk_v1(&raw[ipx0], &cc[ipx0*8], size_blk, &out[ipx0]);
  }
  return duration_us(time_now() - t0).count();
}


time_t calib_jungfrau_v2(const rawd_t *raw, const cc_t *cc, const size_t& size, const size_t& size_blk, out_t *out)
{
  // assuming that
  // * constants are defined as cc[<number-of-pixels>][8],
  //   where 8 stands for for ALL 4 combinations of gain bits for peds then gain.
  //   cc.shape = (<number-of-pixels-in detector>, <2-for-peds-and-gains>, <4-gain-ranges>) = (npix, 2, 4)
  // * raw and out have letgth of size, where
  // size_blk is not used

  time_point_t t0 = time_now();
  size_t igap = 4*size;
  size_t icc;
  for (size_t i=0; i<size; ++i) {
    icc = i + size*(raw[i] >> BSH); // index of calibration constants of V2
    out[i] = ((raw[i] & MDA) - cc[icc]) * cc[icc+igap];
  }
  return duration_us(time_now() - t0).count();
}

}; // namespace utilsdetector

// EOF

