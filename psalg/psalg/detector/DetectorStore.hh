#ifndef PSALG_DETECTORSTORE_H
#define PSALG_DETECTORSTORE_H
//-----------------------------

/** Usage
 *
 * #include "psalg/detector/DetectorStore.hh"
 *
 * Detector* det = getDetector("Cspad.");
 * NDArray<pedestals_t> peds = det->raw();
 * NDArray<pedestals_t> peds = det->pedestals();
 */

#include "psalg/detector/Detector.hh"

namespace detector {

  Detector* getDetector(const std::string& detname);

} // namespace detector

#endif // PSALG_DETECTORSTORE_H
//-----------------------------
