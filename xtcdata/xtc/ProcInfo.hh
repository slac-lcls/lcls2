#ifndef XtcData_ProcInfo_hh
#define XtcData_ProcInfo_hh

#include "xtcdata/xtc/Level.hh"
#include "xtcdata/xtc/Src.hh"
#include <stdint.h>

namespace XtcData
{

// For all levels except Source

class ProcInfo : public Src
{
    public:
    //     ProcInfo();
    ProcInfo(Level::Type level, uint32_t processId, uint32_t ipAddr);

    uint32_t processId() const;
    uint32_t ipAddr() const;
    void ipAddr(int);
};
}
#endif
