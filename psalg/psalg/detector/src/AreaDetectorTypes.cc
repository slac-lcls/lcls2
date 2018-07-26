//-----------------------------

#include "psalg/detector/AreaDetectorTypes.hh"
#include "psalg/utils/Logger.hh" // for MSG, MSGSTREAM
#include <iostream> //ostream, cout

//using namespace detector;

namespace detector {

//-----------------------------

const AREADETTYPE find_area_dettype(const std::string& detname) {
  for(std::map<std::string, AREADETTYPE>::const_iterator 
    it =map_area_detname_to_dettype.begin(); 
    it!=map_area_detname_to_dettype.end(); ++it) {
    if(detname.find(it->first) != std::string::npos) return it->second;
  }
  return UNDEFINED;
}

void print_map_area_detname_to_dettype() {
  MSGSTREAM(INFO, out){
    for (std::map<std::string, AREADETTYPE>::const_iterator 
       it =map_area_detname_to_dettype.begin(); 
       it!=map_area_detname_to_dettype.end(); ++it) {
       out << it->first << " => " << it->second << '\n'; // std::cout
    }
  }
}

} // namespace detector

//-----------------------------
