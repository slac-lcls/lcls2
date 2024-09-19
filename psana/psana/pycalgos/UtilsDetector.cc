
#include "psana/pycalgos/UtilsDetector.hh"
//#include <sstream>   // for stringstream
using namespace std;

//namespace utilsdetector {

//void calib_epix10ka(const fraw_t *raw, const peds_t *peds, const gain_t *gain, const mask_t *mask, const size_t& size, fraw_t *out)
void calib_epix10ka(fraw_t *raw, peds_t *peds, gain_t *gain, mask_t *mask, const size_t& size, fraw_t *out);
{
  for (size_t i=0; i<size; ++i) {
    out[i] = mask[i]>0 ? (raw[i] - peds[i])*gain[i] : 0;
  }
}

  //}; // namespace utilsdetector

// EOF

