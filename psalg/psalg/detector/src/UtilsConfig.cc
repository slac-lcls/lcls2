//-------------------

//#include <iostream> //ostream, cout
#include <sstream>  // for stringstream
#include <stdio.h>  // for  sprintf, printf( "%lf\n", accum );
#include <iomanip>  // for setw, hex, setprecision, right

#include "psalg/detector/UtilsConfig.hh"
#include "psalg/utils/Logger.hh" // for MSG, MSGSTREAM

using namespace std; 
using namespace XtcData;

namespace detector {

//-------------------

  //// extract Names from ConfigIter
  ////NamesId& namesId = configo.value().namesId();
  //NamesId& namesId = configo.shape().namesId();
  //NamesLookup& namesLookup = configo.namesLookup();
  //NameIndex& nameindex = namesLookup[namesId];
  //Names& names = nameindex.names();

  Names& configNames(ConfigIter& configo) {
    return configo.namesLookup()[configo.shape().namesId()].names();
  }

//-------------------

std::string str_dg_info(const Dgram* dg) {
  std::stringstream ss;
  ss << " ==== transition: " << TransitionId::name(dg->service())
     << " of type: "   << dg->type()
     << " time "       << dg->time.seconds() 
     << '.' << std::setw(9) << setfill('0') << dg->time.nanoseconds()
     << " env "        << dg->env
     << " payloadSize "<< dg->xtc.sizeofPayload()
     << " extent "     << dg->xtc.extent;
  return ss.str();
}

//-------------------

void print_dg_info(const Dgram* dg) {
  printf("%s\n",str_dg_info(dg).c_str());
}

//-------------------

std::string str_config_names(XtcData::ConfigIter& configo) {
  XtcData::NamesId& namesId = configo.shape().namesId();
  XtcData::Names& names = configNames(configo);
  std::stringstream ss;
  ss  << "UtilsConfig::str_config_names" 
      << " transition: " << namesId.namesId() << " (0/1 = config/data)\n"
      << " Names:: detName: " << names.detName() 
      << " detType: " << names.detType() 
      << " detId: " << names.detId()
      << " segment: " << names.segment() 
      << " number of names: " << names.num()
      << " alg.name: " << names.alg().name();
  return ss.str();
}

//-------------------

void print_config_names(XtcData::ConfigIter& configo) {
  printf(str_config_names(configo).c_str());
}

//-------------------

} // namespace detector

//-------------------
