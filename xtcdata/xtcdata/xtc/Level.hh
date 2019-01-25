#ifndef PDSLEVEL_HH
#define PDSLEVEL_HH

namespace XtcData
{
class Level
{
public:
    enum Type { Segment, Event, NumberOfLevels };
    static const char* name(Type type);
};
}

#endif
