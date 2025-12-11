#ifndef PSALG_UTILSCONFIG_H
#define PSALG_UTILSCONFIG_H

//-------------------

/** Usage
 * #include "UtilsConfig.hh"
 */

//#include <string>
//#include <map>

//#include "NDArray.hh" // NDArray
//#include "AreaDetectorTypes.hh"

#include "ConfigIter.hh"

//#include "Dgram.hh"
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
