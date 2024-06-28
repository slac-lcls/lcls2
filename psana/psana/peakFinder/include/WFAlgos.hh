#ifndef PSALG_PEAKS_WFALGOS_H
#define PSALG_PEAKS_WFALGOS_H

#include <vector>
#include <cstdint>  // uint32_t

//#include "psalg/calib/NDArray.hh"
//#include <utility>  // pair
//#include "psalg/utils/Utils.hh" // Pair
//typedef Pair<wfdata_t, index_t> Edge;
//std::vector< std::pair<T, index_t> >& result);

  /**
   * @ingroup EdgeFinder
   *
   * @brief Waveform pulse edge finder
   *
   * Generates an array of hit times and amplitudes for waveform
   * leading (trailing) edges using a constant fraction discriminator
   * algorithm.  The baseline and minimum amplitude threshold are used
   * for discriminating hits.  The pulse height fraction at which the hit
   * time is derived is also required as input.  Note that if the threshold
   * is less than the baseline value, then leading edges are "falling" and
   * trailing edges are "rising".  In order for two pulses to be discriminated,
   * the waveform samples below the two pulses must fall below (or above for
   * negative pulses) the fractional value of the threshold; i.e.
   * waveform[i] < fraction*(threshold-baseline)+baseline.
   *
   * The results are stored in a 2D array such that result[i][0] is the time
   * (waveform sample) of the i'th hit and result[i][1] is the maximum amplitude
   * of the i'th hit.
   *
   */

//using namespace psalg;

namespace psalg {

typedef uint32_t index_t;
typedef double wfdata_t;

template <typename T>
void
_add_edge(
  const std::vector<T>& v,
  bool     rising,
  double   fraction,
  double   deadtime,
  T        peak,
  index_t  start,
  double&  last,
  index_t& ipk,
  T*       pkvals,
  index_t* pkinds);

template <typename T>
index_t
find_edges(
  index_t  npkmax,
  T*       pkvals,
  index_t* pkinds,
  const std::vector<T>& wf,
  double   baseline,
  double   threshold,
  double   fraction,
  double   deadtime,
  bool     leading_edge
);

} // namespace psalg

#endif // PSALG_PEAKS_WFALGOS_H
