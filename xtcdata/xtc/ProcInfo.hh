#ifndef Pds_ProcInfo_hh
#define Pds_ProcInfo_hh

#include "pdsdata/xtc/Level.hh"
#include "pdsdata/xtc/Src.hh"
#include <stdint.h>

namespace Pds
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
