#ifndef XtcData_TransitionId_hh
#define XtcData_TransitionId_hh

namespace XtcData
{

class TransitionId
{
public:
    enum Value {
        Unknown,
        Reset,
        Configure,
        Unconfigure,
        Enable,
        Disable,
        ConfigUpdate,
        BeginRecord,
        EndRecord,
        Unused_09,
        Unused_10,
        Unused_11,
        L1Accept = 12,                  // Must be 12
        NumberOf
    };
    static const char* name(TransitionId::Value id);
};
}

#endif
