#ifndef XtcData_Src_hh
#define XtcData_Src_hh

#include "xtcdata/xtc/Level.hh"
#include <stdint.h>

namespace XtcData
{

class Node;

class Src
{
    public:
    Src();
    Src(Level::Type level);

    uint32_t log() const;
    uint32_t phy() const;

    Level::Type level() const;

    bool operator==(const Src& s) const;
    bool operator<(const Src& s) const;

    static uint32_t _sizeof()
    {
        return sizeof(Src);
    }

    protected:
    uint32_t _log; // logical  identifier
    uint32_t _phy; // physical identifier
};
}
#endif
