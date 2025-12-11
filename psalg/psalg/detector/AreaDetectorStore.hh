#ifndef PSALG_AREADETECTORSTORE_H
#define PSALG_AREADETECTORSTORE_H
//-----------------------------

/** Usage
 *
 * #include "AreaDetectorStore.hh"
 *
 * AreaDetector* det = getAreaDetector("Cspad.");
 * NDArray<pedestals_t> peds = det->raw();
 * NDArray<pedestals_t> peds = det->pedestals();
 */

#include "AreaDetector.hh"
#include "ConfigIter.hh"

namespace detector {

  //class AreaDetector;

  // returns pointer to AreaDetector to access raw data and calibration parameters.
  AreaDetector* getAreaDetector(const std::string& detname, XtcData::ConfigIter& configo);

  // returns pointer to AreaDetector to ACCESS CALIBRATION PARS ONLY !
  AreaDetector* getAreaDetector(const std::string& detname);

} // namespace detector

#endif // PSALG_AREADETECTORSTORE_H
//-----------------------------
