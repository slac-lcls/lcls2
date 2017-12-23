#ifndef XtcData_Dgram_hh
#define XtcData_Dgram_hh

#include "Sequence.hh"
#include "Xtc.hh"
#include <stdint.h>

#pragma pack(push,4)

namespace XtcData
{

#define PDS_DGRAM_STRUCT   \
    Sequence seq;          \
    unsigned evtcounter:24;\
    unsigned version:8;    \
    uint64_t env;          \
    Xtc      xtc

class Dgram
{
public:
    PDS_DGRAM_STRUCT;
};
}

#pragma pack(pop)

#endif
