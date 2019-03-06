//-------------------

#include "psalg/detector/UtilsConfig.hh"
#include "psalg/utils/Logger.hh" // for MSG, MSGSTREAM
//#include <iostream> //ostream, cout

#include <stdio.h> // for  sprintf, printf( "%lf\n", accum );
//using namespace detector;

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

void print_dg_info(const Dgram* dg) {
      printf(" ==== transition: %s of type: %d time %d.%09d, pulseId %lux, env %ux, "
             "payloadSize %d extent %d\n",
             TransitionId::name(dg->seq.service()), dg->seq.type(), dg->seq.stamp().seconds(),
             dg->seq.stamp().nanoseconds(), dg->seq.pulseId().value(),
             dg->env, dg->xtc.sizeofPayload(), dg->xtc.extent);
}

//-------------------

} // namespace detector

//-------------------
