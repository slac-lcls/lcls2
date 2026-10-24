/**
 * @file
 * @brief Declares XtcData::Src, the 32-bit source word (level + value) stored in each Xtc header.
 */
#ifndef XtcData_Src_hh
#define XtcData_Src_hh

#include <limits>
#include <stdint.h>
#include "xtcdata/xtc/Level.hh"

namespace XtcData
{

/** 32-bit source word: bits 31:28 hold a Level::Type and bits 27:0 a value. */
class Src
{
private:
    enum { LevelBitMask = 0xf0000000, LevelBitShift = 28 };
    enum { ValueBitMask = 0x0fffffff };

public:
    /** Set the level bits to level and the value bits to 0. */
    Src(Level::Type level=Level::Segment) :
        _value(level<<LevelBitShift) {}
    /** Store value & 0x0fffffff in bits 27:0 and level in bits 31:28. */
    Src(unsigned value, Level::Type level=Level::Segment) :
        _value((value&ValueBitMask)|((level<<LevelBitShift)&LevelBitMask)) {}

    /** Return bits 31:28 as a Level::Type. */
    Level::Type level() const {return (Level::Type)((_value&LevelBitMask)>>LevelBitShift);}
    /** Return bits 27:0. */
    unsigned    value() const {return _value&ValueBitMask;}

  protected:
    uint32_t _value;
};

}

#endif
