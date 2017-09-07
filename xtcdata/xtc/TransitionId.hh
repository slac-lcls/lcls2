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
        Map,
        Unmap,
        Configure,
        Unconfigure,
        BeginRun,
        EndRun,
        BeginCalibCycle,
        EndCalibCycle,
        Enable,
        Disable,
        L1Accept,
        NumberOf
    };
    static const char* name(TransitionId::Value id);
};
}

#endif
