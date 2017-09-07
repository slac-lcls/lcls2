#ifndef XtcData_Dgram_hh
#define XtcData_Dgram_hh

#include "Env.hh"
#include "Sequence.hh"
#include "Xtc.hh"

namespace XtcData
{

#define PDS_DGRAM_STRUCT \
    Sequence seq;        \
    Env env;             \
    Xtc xtc

class Dgram
{
    public:
    PDS_DGRAM_STRUCT;
};
}
#endif
