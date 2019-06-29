#include "xtcdata/xtc/TransitionId.hh"

using namespace XtcData;

const char* TransitionId::name(TransitionId::Value id)
{
    static const char* _names[] = {
        "Unknown",
        "Reset",
        "Configure",
        "Unconfigure",
        "Enable",
        "Disable",
        "ConfigUpdate",
        "BeginRecord",
        "EndRecord",
        "SlowUpdate",
        "Unused_10",
        "Unused_11",
        "L1Accept",
    };
    return (id < TransitionId::NumberOf ? _names[id] : "-Invalid-");
};
