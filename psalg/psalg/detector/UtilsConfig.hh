/**
 * @file
 * @brief Declares helpers that print or return descriptions of a ConfigIter's Names and of a Dgram header.
 */
#ifndef PSALG_UTILSCONFIG_H
#define PSALG_UTILSCONFIG_H

//-------------------

/** Usage
 * #include "psalg/detector/UtilsConfig.hh"
 */

//#include <string>
//#include <map>

//#include "psalg/calib/NDArray.hh" // NDArray
//#include "psalg/calib/AreaDetectorTypes.hh"

#include "xtcdata/xtc/ConfigIter.hh"

//#include "xtcdata/xtc/Dgram.hh"
//typedef XtcData::ConfigIter ConfigIter;

//-------------------
//using namespace std; 
using namespace XtcData;  // this is evil

namespace detector {

  /** Return the Names that configo's NamesLookup holds for the NamesId of configo.shape(); NameIndex::names() throws if there is none. */
  XtcData::Names& configNames(XtcData::ConfigIter& configo);

  /** Return a one-line description of dg: transition name, type, time (seconds.nanoseconds), env, payload size and extent. */
  std::string str_dg_info(const XtcData::Dgram* dg);
  /** Print str_dg_info(dg) followed by a newline to stdout. */
  void      print_dg_info(const XtcData::Dgram* dg);

  /** Return a description of configo.shape()'s NamesId and Names: names id, detName, detType, detId, segment, number of names and alg name. */
  std::string str_config_names(XtcData::ConfigIter& configo);
  /** Print str_config_names(configo) to stdout; the string is passed to printf as the format string and no newline is added. */
  void      print_config_names(XtcData::ConfigIter& configo);

} // namespace detector

//-------------------

#endif // PSALG_UTILSCONFIG_H
