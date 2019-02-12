
#include "ProcInfo.hh"

using namespace psalg::shmem;

ProcInfo::ProcInfo(Level::Type level, uint32_t processId, uint32_t ipAddr) :
  Src(level) {
  }

uint32_t ProcInfo::processId()    const {return 0;}
uint32_t ProcInfo::ipAddr()       const {return 0;}
void     ProcInfo::ipAddr(int ip)       {return;}
