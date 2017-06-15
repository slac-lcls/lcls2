#include "pdsdata/xtc/Level.hh"

using namespace Pds;

const char* Level::name(Type type)
{ 
  static const char* _names[] = {
    "Control",
    "Source",
    "Segment",
    "Event",
    "Recorder",
    "Observer",
    "Reporter"
  };
  return (type < NumberOfLevels ? _names[type] : "-Invalid-");
}
