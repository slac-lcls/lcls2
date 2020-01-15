#ifndef XtcData_Smd_hh
#define XtcData_Smd_hh

#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/DescData.hh"

namespace XtcData
{

class Smd
{
public:

    Smd() {
    };

    Dgram* generate(Dgram* dgIn, void* buf, uint64_t offset, uint64_t size,
                    NamesLookup& namesLookup, NamesId namesId);

}; // end class Smd

}; // end namespace XtcData


#endif
