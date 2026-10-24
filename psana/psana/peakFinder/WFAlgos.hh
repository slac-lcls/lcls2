/**
 * @file
 * @brief Declares the waveform edge finder psalg::find_edges() and its helper _add_edge().
 */
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

/** uint32_t; index type of the find_edges() results. */
typedef uint32_t index_t;
/** double. Not used in WFAlgos.hh or WFAlgos.cc. */
typedef double wfdata_t;

/**
 * From sample start, find the first sample i where v reaches the level fraction (an absolute level: upward if rising, downward otherwise) and interpolate the edge position linearly between samples i-1 and i (0 if i is 0).
 * If no edge was recorded yet (last < 0) or the edge is more than deadtime after last, store peak in pkvals[ipk] and the edge, truncated to index_t, in pkinds[ipk], then increment ipk and set last. The search is not bounded by the size of v.
 */
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

/**
 * Find pulses in wf that go beyond threshold (above it if threshold > baseline, else below) for more than deadtime samples; for each, store its extreme value in pkvals and the interpolated index of the edge at level fraction*(peak-baseline)+baseline in pkinds (leading edge, or with leading_edge false the trailing edge searched from the peak).
 * Edges within deadtime of the previous one are skipped and at most npkmax peaks are stored. Instantiated in WFAlgos.cc for double, float, int, int64_t and int16_t.
 * @return The number of peaks stored.
 */
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
