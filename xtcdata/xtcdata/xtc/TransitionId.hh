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
        L1Accept,
        ConfigUpdate,
        BeginRecord,
        EndRecord,
        NumberOf
    };
    static const char* name(TransitionId::Value id);
};
}

#endif
