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

/*
void print_dg_info(const Dgram* dg) {
      printf(" ==== transition: %s of type: %d time %d.%09d, pulseId %lux, env %ux, "
             "payloadSize %d extent %d\n",
             TransitionId::name(dg->seq.service()), dg->seq.type(), dg->seq.stamp().seconds(),
             dg->seq.stamp().nanoseconds(), dg->seq.pulseId().value(),
             dg->env, dg->xtc.sizeofPayload(), dg->xtc.extent);
}
*/

//-------------------

std::string str_dg_info(const Dgram* dg) {
  std::stringstream ss;
  ss << " ==== transition: " << TransitionId::name(dg->seq.service())
     << " of type: "   << dg->seq.type()
     << " time "       << dg->seq.stamp().seconds() 
     << '.' << std::setw(9) << setfill('0') << dg->seq.stamp().nanoseconds()
     << " pulseId "    << dg->seq.pulseId().value()
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
