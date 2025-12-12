
#include "pycalgos/UtilsDetector.hh"
#include <iostream> // std::cout
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
//time_t calib_std_2(rawd_t *raw, peds_t *peds, gain_t *gain, mask_t *mask, const sizeb_t& size, const rawd_t databits, out_t *out)
time_t calib_std_2(const rawd_t *raw, const peds_t *peds, const gain_t *gain, const mask_t *mask, const sizeb_t& size, const rawd_t databits, out_t *out)
{
  time_point_t t0 = time_now();
  for (sizeb_t i=0; i<size; ++i) {
    //out[i] = mask[i]>0 ? ((raw[i] & databits) - peds[i])*gain[i] : 0;
    out[i] = ((raw[i] & databits) - peds[i])*gain[i]*mask[i];
  }
  return duration_us(time_now() - t0).count();
}

//time_t calib_std_v2(rawd_t *raw, peds_t *peds, gain_t *gain, mask_t *mask, const sizeb_t& size, const rawd_t databits, out_t *out)
time_t calib_std_v3(const rawd_t *raw, const peds_t *peds, const gain_t *gain, const mask_t *mask, const sizeb_t& size, const rawd_t databits, out_t *out)
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


time_t calib_std(const rawd_t *raw, const peds_t *peds, const gain_t *gain, const mask_t *mask, const sizeb_t& size, const rawd_t databits, out_t *out)
{
  time_point_t t0 = time_now();
  const rawd_t *r = raw;
  const peds_t *p = peds;
  const gain_t *g = gain;
  const mask_t *m = mask;
  out_t *o = out;
  while (r<raw+size) {
    *o++ = ((*r++ & databits) - *p++)*(*g++)*(*m++);
  }
  return duration_us(time_now() - t0).count();
}


time_t calib_jungfrau_v0(const rawd_t *raw, const peds_t *peds, const gain_t *gain, const mask_t *mask, const sizeb_t& size, out_t *out)
{
  //raw[i], out[i], mask[i] shape:   (<number-of-segs>, 512, 1024)
  //peds[icc], gain[icc]    shape:(3, <number-of-segs>, 512, 1024)
  time_point_t t0 = time_now();
  rawd_t rawi;  // copy of raw[i]
  uint8_t igr;  // index of the gain range: 0,1,2
  sizeb_t icc; // pixel index in the array of calibration constants
  for (sizeb_t i=0; i<size; ++i) {
    rawi = raw[i];
    igr = rawi >> BSH; if (igr > 1) igr=2; // gain bits gr0/1/2 = 00/01/11. Combination 10 - bad pixel accounted in pixel_status
    icc = igr * size + i;
    out[i] = ((rawi & MDA) - peds[icc])*gain[icc]*mask[i];
  }
  return duration_us(time_now() - t0).count();
}


void calib_jungrfau_blk_v1(const rawd_t *raw, const cc_t *cc, const sizeb_t& size_blk, out_t *out)
{
  //float cc[size_block][8], where 8 is 4-peds and 4 gains per pixel
  //uint8_t igr; // index of the gain range index gr0/1/2 = 00/01/11, combination 10 - bad pixel status
  sizeb_t icc;
  for (sizeb_t i=0; i<size_blk; ++i) {
    icc = 8*i + (raw[i] >> BSH); // index of calibration constants 4*granges for peds and gains
    out[i] = ((raw[i] & MDA) - cc[icc]) * cc[icc+4];
  }
}

  //void calib_jungrfau_blk_v2(const rawd_t *raw, const ccstruct *cc, const sizeb_t& size_blk, out_t *out)
  //{
  //  ccstruct& ccgrpix;
  //  for (sizeb_t i=0; i<size_blk; ++i) {
  //    ccgrpix = cc[i][rawi >> BSH]; // gain range index gr0/1/2 = 00/01/11, combination 10 - bad pixel status
  //    out[i] = (raw[i] & MDA - ccgrpix.pedestal) * ccgrpix.gain;
  //  }
  //}


time_t calib_jungfrau_v1(const rawd_t *raw, const cc_t *cc, const sizeb_t& size, const sizeb_t& size_blk, out_t *out)
{
  // V1 - assuming that
  // * constants are defined as cc[<number-of-pixels>][8],
  //   where 8 stands for for ALL 4 combinations of gain bits for peds then gain.
  //   cc.shape = (<number-of-pixels-in detector>, <2-for-peds-and-gains>, <4-gain-ranges>) = (npix, 2, 4)
  // * raw and out have letgth of size, where
  //   size = size_blk * (int)number_of_blocks
  time_point_t t0 = time_now();
  sizeb_t ipx0; // the 0-th pixel index of the block in data and other arrays
  for (sizeb_t ib=0; ib<size/size_blk; ++ib) { // loop over blocks in entire data array of size
    ipx0 = ib*size_blk;
    calib_jungrfau_blk_v1(&raw[ipx0], &cc[ipx0*8], size_blk, &out[ipx0]);
  }
  return duration_us(time_now() - t0).count();
}


time_t calib_jungfrau_v2(const rawd_t *raw, const cc_t *cc, const sizeb_t& size, const sizeb_t& size_blk, out_t *out)
{
  // V2 - assuming that
  // * constants are defined as cc[8][<number-of-pixels>],
  //   where 8 stands for ALL 4 combinations of gain bits for peds then gain.
  //   cc.shape = (<2-for-peds-and-gains>, <4-gain-ranges>, <number-of-pixels-in detector>) = (2, 4, npix)
  // * raw and out have letgth of size, where
  // size_blk IS NOT USED
  time_point_t t0 = time_now();
  sizeb_t igap = 4*size;
  sizeb_t icc;
  for (sizeb_t i=0; i<size; ++i) {
    icc = i + size*(raw[i] >> BSH); // index of calibration constants of V2
    out[i] = ((raw[i] & MDA) - cc[icc]) * cc[icc+igap];
  }
  return duration_us(time_now() - t0).count();
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
    icc = 2*(i + size*(rawt >> BSH)); // index of calibration constants of V3
    //icc = 2*(i + size*((rawt >> BSH) & 0x3)); // index of calibration constants of V3
    //std::cout << "  peds-offset:" << cc[icc] << " gain:" << cc[icc+1] << std::endl;
    out[i] = ((rawt & MDA) - cc[icc]) * cc[icc+1];
  }
  return duration_us(time_now() - t0).count();
}


time_t calib_jungfrau_v3_struct(const rawd_t *raw, const cc_t *cc, const sizeb_t& size, const sizeb_t& size_blk, out_t *out)
{
  // V3 - use struct,  assuming that
  // * constants are defined as cc[4][<number-of-pixels>][2] - Rick's shape,
  //   where 4 stands for combinations of gain bits, 00,01,10,11, and 2 for peds-offset then gain*mask.
  //   cc.shape = (<4-gain-ranges>, <number-of-pixels-in detector>, <2-for-peds-and-gains>) = (4, npix, 2)
  // * raw and out have letgth of size, where
  // size_blk IS NOT USED

  // Ric?s code structure proposal:
  // > pg[4][N]
  // > for i in range(N):
  // >  gain=data[i]>>14 & 0x3 # valid gain is 0,1,3.  gain 2 should have pg.gain 0
  // >  datum = data[i] & 0x3fff
  // >  result = (datam-pg[gain][i].pedestal)*pg[gain][i].gain
  time_point_t t0 = time_now();
  fill_CCSV3(cc);
  ccstruct *pcc;
  for (sizeb_t i=0; i<size; ++i) {
    pcc = &CCSV3[raw[i] >> BSH][i];
    out[i] = ((raw[i] & MDA) - pcc->pedestal) * pcc->gain;
  }
  return duration_us(time_now() - t0).count();
}


void fill_CCSV3(const cc_t *cc)
{ // fills  ccstruct CCSV3[4][16777216];
  static int counter = -1; counter++;
  //std::cout << "  counter:" << counter << std::endl;
  if (counter>0) return;
  std::cout << "fill_CCSV3 at 1st entrance only" << std::endl;
  sizeb_t icc;
  for (int n=0; n<NGRINDS; n++) {
    std::cout << "gain bits combination: " << n << std::endl;
    for (int i=0; i<NPIXELS; i++) {
      icc = 2*(i + NPIXELS*n);
      CCSV3[n][i].pedestal = cc[icc];
      CCSV3[n][i].gain     = cc[icc+1];
      if (i<5) {
	std::cout << "pixel: " << i << " pedestal: " <<  CCSV3[n][i].pedestal << " gain: " << CCSV3[n][i].gain << std::endl;
      }
    }
  }
  std::cout << "fill_CCSV3 initialization of CCSV3 is completed" << std::endl;
}


time_t calib_jungfrau_v4_empty()
{
  time_point_t t0 = time_now();
  return duration_us(time_now() - t0).count();
}


time_t calib_jungfrau_v5_empty(const rawd_t *raw, const cc_t *cc, const sizeb_t& size, const sizeb_t& size_blk, out_t *out)
{
  return calib_jungfrau_v4_empty();
}


CalibConsSingleton* CalibConsSingleton::m_pInstance = NULL; // !!!!!! make global pointer !!!!!
CalibConsSingleton::CalibConsSingleton()
{
  std::cout << "!!!!!!!! Single instance for singleton class CalibConsSingleton is created \n";
}
CalibConsSingleton* CalibConsSingleton::instance()
{
  if(!m_pInstance) m_pInstance = new CalibConsSingleton();
  return m_pInstance;
}
void CalibConsSingleton::print() {std::cout << "CalibConsSingleton::print()\n";}


}; // namespace utilsdetector

// EOF

