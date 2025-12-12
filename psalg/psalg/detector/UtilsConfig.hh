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

  XtcData::Names& configNames(XtcData::ConfigIter& configo);

  std::string str_dg_info(const XtcData::Dgram* dg);
  void      print_dg_info(const XtcData::Dgram* dg);

  std::string str_config_names(XtcData::ConfigIter& configo);
  void      print_config_names(XtcData::ConfigIter& configo);

} // namespace detector

//-------------------

#endif // PSALG_UTILSCONFIG_H
