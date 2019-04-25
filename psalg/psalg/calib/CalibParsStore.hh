#ifndef PSALG_CALIBPARSSTORE_H
#define PSALG_CALIBPARSSTORE_H
//-----------------------------

/** Usage
 *
 * #include "psalg/calib/CalibParsStore.hh"
 * query_t query = 123;
 * CalibPars* cp = getCalibPars("Epix100a");
 * NDArray<pedestals_t>& peds = cp->pedestals(query);
 * NDArray<common_mode_t>& cmode = cp->common_mode(query);
 * geometry_t& strgeo = cp->geometry();
 */

#include "psalg/calib/CalibPars.hh"

namespace calib {

  CalibPars* getCalibPars(const char* detname = "undefined");

} // namespace calib

#endif // PSALG_CALIBPARSSTORE_H
//-----------------------------
