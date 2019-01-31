#ifndef XtcData_Src_hh
#define XtcData_Src_hh

#include <limits>
#include <stdint.h>
#include "xtcdata/xtc/Level.hh"

namespace XtcData
{

class Src
{
private:
    enum { LevelBitMask = 0xf0000000, LevelBitShift = 28 };
    enum { ValueBitMask = 0x0fffffff };

public:
    Src(Level::Type level=Level::Segment) :
        _value(level<<LevelBitShift) {}
    Src(unsigned value, Level::Type level=Level::Segment) :
        _value((value&ValueBitMask)|((level<<LevelBitShift)&LevelBitMask)) {}

    Level::Type level() const {return (Level::Type)((_value&LevelBitMask)>>LevelBitShift);}
    unsigned    value() const {return _value&ValueBitMask;}

  protected:
    uint32_t _value;
};

}

#endif
