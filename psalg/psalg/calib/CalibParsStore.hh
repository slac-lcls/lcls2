#ifndef PSALG_CALIBPARSSTORE_H
#define PSALG_CALIBPARSSTORE_H
//-----------------------------

/** Usage
 *
 * #include "psalg/calib/CalibParsStore.hh"
 *
 * CalibPars* cp = getAreaDetector("Cspad.");
 * NDArray<pedestals_t> peds = det->pedestals();
 * NDArray<common_mode_t> cmode = det->common_mode();
 * geometry_t& strgeo = cp->geometry();
 */

#include "psalg/calib/CalibPars.hh"

namespace calib {

  CalibPars* getCalibPars(const std::string& detname);

} // namespace calib

#endif // PSALG_CALIBPARSSTORE_H
//-----------------------------
