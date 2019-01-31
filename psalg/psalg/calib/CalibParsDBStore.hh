#ifndef PSALG_CALIBPARSDBSTORE_H
#define PSALG_CALIBPARSDBSTORE_H
//-----------------------------

/** Factory for DB type selection.
 *
 * Usage
 *
 * #include "psalg/calib/CalibParsDBStore.hh"
 * query_t query = 123;
 * CalibPars* cp = getCalibParsDB("epix100a-", calib::DBWEB);
 * NDArray<pedestals_t>& peds = cp->pedestals(query);
 * NDArray<common_mode_t>& cmode = cp->common_mode(query);
 * geometry_t& strgeo = cp->geometry();
 */

#include "psalg/calib/CalibPars.hh"

namespace calib {

  enum DBTYPE {DBWEB=0, DBMONGO, DBCALIB, DBHDF5};

  CalibPars* getCalibParsDB(const std::string& detname, const DBTYPE& dbtype=calib::DBWEB);

} // namespace calib

#endif // PSALG_CALIBPARSDBSTORE_H
//-----------------------------
