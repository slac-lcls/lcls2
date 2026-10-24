/**
 * @file
 * @brief Declares XtcData::Level, an enum of levels (Segment, Event) with a name lookup.
 */
#ifndef PDSLEVEL_HH
#define PDSLEVEL_HH

namespace XtcData
{
/** Wrapper class for the Level::Type enum and its name() lookup. */
class Level
{
public:
    /** Level identifiers. */
    enum Type { Segment, /**< Value 0; name() returns "Segment". */ Event, /**< Value 1; name() returns "Event". */ NumberOfLevels  /**< Value 2, the number of levels; name() returns "-Invalid-" for values at or above it. */ };
    /** Return "Segment" or "Event" for type, or "-Invalid-" if type >= NumberOfLevels. */
    static const char* name(Type type);
};
}

#endif
