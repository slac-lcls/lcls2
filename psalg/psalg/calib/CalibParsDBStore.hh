#ifndef PSALG_CALIBPARSDBSTORE_H
#define PSALG_CALIBPARSDBSTORE_H
//-----------------------------

/** Factory for DB type selection.
 *
 * Usage::
 *
 * #include "psalg/calib/CalibParsDBStore.hh"
 *
 * Query q("some-string defining DB, derector, collection, constants type, run or timestamp is here");
 *
 * CalibParsDB* o = new getCalibParsDB(calib::DBVEB);
 * std::cout << "In test_CalibParsDB dbtypename: " << o->dbtypename() << '\n';
 *
 * const NDArray<float>&     nda_float  = o->get_ndarray_float(q);
 * const NDArray<double>&    nda_double = o->get_ndarray_double(q);
 * const NDArray<uint16_t>&  nda_uint16 = o->get_ndarray_uint16(q); 
 * const NDArray<uint32_t>&  nda_uint32 = o->get_ndarray_uint32(q); 
 * std::string&              s          = o->get_string(q);
 */
 
#include "psalg/calib/CalibParsDB.hh"

namespace calib {

  enum DBTYPE {DBDEF, DBWEB, DBMONGO, DBCALIB, DBHDF5};

  CalibParsDB* getCalibParsDB(const DBTYPE& dbtype=calib::DBWEB);

} // namespace calib

#endif // PSALG_CALIBPARSDBSTORE_H
//-----------------------------
