#ifndef PSALG_PEAKS_WFALGOS_H
#define PSALG_PEAKS_WFALGOS_H

#include "psalg/calib/NDArray.hh"
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
   * waveform[i] < fraction*(threshold+baseline).
   *
   * The results are stored in a 2D array such that result[i][0] is the time 
   * (waveform sample) of the i'th hit and result[i][1] is the maximum amplitude 
   * of the i'th hit.
   *
   */

//using namespace psalg;

namespace psalg {

NDArray<double>*
find_edges(NDArray<const double>& waveform,
           double baseline,
           double threshold,
           double fraction=0.5,
           double deadtime=0,
           bool   leading_edges=true);

} // namespace psalg

#endif // PSALG_PEAKS_WFALGOS_H
