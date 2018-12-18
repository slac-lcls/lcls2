#ifndef XtcData_Src_hh
#define XtcData_Src_hh

#include <limits>
#include <stdint.h>
#include "xtcdata/xtc/Level.hh"

namespace XtcData
{

class Src
{
public:
    Src() : _value(std::numeric_limits<uint32_t>::max()) {}
    Src(unsigned value) : _value(value) {}

    unsigned             value() const {return _value&0xffffff;}

  protected:
    uint32_t _value;
};

}

#endif
