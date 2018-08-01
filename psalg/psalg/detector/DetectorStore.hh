#ifndef PSALG_DETECTORSTORE_H
#define PSALG_DETECTORSTORE_H
//-----------------------------

/** Usage
 *
 * #include "psalg/detector/DetectorStore.hh"
 *
 * Detector* det = getDetector("Cspad."); // however methods of AreaDetector will not be accessible...
 * AreaDetector* area_det = dynamic_cast<AreaDetector*>(getDetector("Epix100a"));
 * NDArray<pedestals_t> peds = area_det->raw();
 * NDArray<pedestals_t> peds = area_det->pedestals();
 *
 * #include <iostream> // cout, puts etc.
 * std::cout << "Detector obj: " << *det << '\n';
 * std::cout << "AreaDetector obj: " << *area_det << '\n';
 */

#include "psalg/detector/Detector.hh"

namespace detector {

  Detector* getDetector(const std::string& detname);

} // namespace detector

#endif // PSALG_DETECTORSTORE_H
//-----------------------------
