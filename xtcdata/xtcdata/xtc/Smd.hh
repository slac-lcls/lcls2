#ifndef XtcData_Smd_hh
#define XtcData_Smd_hh

#include "Dgram.hh"
#include "DescData.hh"

namespace XtcData
{

class Smd
{
public:

    Smd() {
    };

    Dgram* generate(Dgram* dgIn, void* buf, const void* bufEnd, uint64_t offset, uint64_t size,
                    NamesLookup& namesLookup, NamesId namesId);

}; // end class Smd

}; // end namespace XtcData


#endif
