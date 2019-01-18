#include "xtcdata/xtc/Level.hh"

using namespace XtcData;

const char* Level::name(Type type)
{
    static const char* _names[] = { "Segment", "Event" };
    return (type < NumberOfLevels ? _names[type] : "-Invalid-");
}
