#ifndef PSALG_AREADETECTORSTORE_H
#define PSALG_AREADETECTORSTORE_H
//-----------------------------

/** Usage
 *
 * #include "psalg/detector/AreaDetectorStore.hh"
 *
 * AreaDetector* det = getAreaDetector("Cspad.");
 * NDArray<pedestals_t> peds = det->raw();
 * NDArray<pedestals_t> peds = det->pedestals();
 */

#include "psalg/detector/AreaDetector.hh"

namespace detector {

  AreaDetector* getAreaDetector(const std::string& detname);

} // namespace detector

#endif // PSALG_AREADETECTORSTORE_H
//-----------------------------
