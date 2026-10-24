/**
 * @file
 * @brief Declares calib::getCalibPars(), the factory for CalibPars objects.
 */
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

  /**
   * Return a new CalibPars for detname: CalibParsEpix100a if find_area_dettype(detname) is EPIX100A, the base CalibPars for UNDEFINED or PNCCD.
   * For any other type it logs a WARNING and throws a const char*. The caller owns the object.
   */
  CalibPars* getCalibPars(const char* detname = "undefined", const DBTYPE& dbtype=DBWEB);

} // namespace calib

#endif // PSALG_CALIBPARSSTORE_H
//-----------------------------
