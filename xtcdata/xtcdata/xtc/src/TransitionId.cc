#include "xtcdata/xtc/TransitionId.hh"

using namespace XtcData;

const char* TransitionId::name(TransitionId::Value id)
{
    static const char* _names[] = {
        "ClearReadout",
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

    // Bail on compilation if someone forgets to update this list
    static_assert(sizeof(_names) / sizeof(*_names) == TransitionId::NumberOf);
};
