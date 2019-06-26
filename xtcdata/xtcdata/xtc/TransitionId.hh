#ifndef XtcData_TransitionId_hh
#define XtcData_TransitionId_hh

namespace XtcData
{

class TransitionId
{
public:
    enum Value {
        // Must keep in synch with strings in src/TransitionId.cc
        // There is also math on these transition id numbers
        // in XtcMonitorServer.cc that assumes they come in pairs.
        // the ConfigUpdate currently breaks this assumption.
        ClearReadout,
        Reset,
        Configure,
        Unconfigure,
        Enable,
        Disable,
        ConfigUpdate,
        BeginRecord,
        EndRecord,
        EpicsUpdate,
        Unused_10,
        Unused_11,
        L1Accept = 12,                  // Must be 12
        NumberOf
    };
    static const char* name(TransitionId::Value id);
};
}

#endif
