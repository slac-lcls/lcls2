//-----------------------------

#include "psalg/detector/DetectorTypes.hh"
#include "psalg/utils/Logger.hh" // for MSG, MSGSTREAM
#include <iostream> //ostream, cout

//using namespace detector;

namespace detector {

//-----------------------------

const DETTYPE find_dettype(const std::string& detname) {
  for(std::map<std::string, DETTYPE>::const_iterator 
    it =map_detname_to_dettype.begin(); 
    it!=map_detname_to_dettype.end(); ++it) {
    if(detname.find(it->first) != std::string::npos) return it->second;
  }
  for(std::map<std::string, DETTYPE>::const_iterator 
    it =map_bldinfo_to_dettype.begin(); 
    it!=map_bldinfo_to_dettype.end(); ++it) {
    if(detname.find(it->first) != std::string::npos) return it->second;
  }
  return UNDEFINED_DETECTOR;
}

void print_map_detname_to_dettype() {
  MSGSTREAM(INFO, out){
    for (std::map<std::string, DETTYPE>::const_iterator 
       it =map_detname_to_dettype.begin(); 
       it!=map_detname_to_dettype.end(); ++it) {
       out << it->first << " => " << it->second << '\n'; // std::cout
    }
  }
}

void print_map_bldinfo_to_dettype() {
  MSGSTREAM(INFO, out){
    for (std::map<std::string, DETTYPE>::const_iterator 
       it =map_bldinfo_to_dettype.begin(); 
       it!=map_bldinfo_to_dettype.end(); ++it) {
       out << it->first << " => " << it->second << '\n'; // std::cout
    }
  }
}

} // namespace detector

//-----------------------------
