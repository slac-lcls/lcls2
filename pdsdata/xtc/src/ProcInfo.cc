
#include "pdsdata/xtc/ProcInfo.hh"

using namespace Pds;

ProcInfo::ProcInfo(Level::Type level, uint32_t processId, uint32_t ipAddr) :
  Src(level) {
  _phy=ipAddr;
  _log|=processId&0x00ffffff;
  }

uint32_t ProcInfo::processId()    const {return _log&0xffffff;}
uint32_t ProcInfo::ipAddr()       const {return _phy;}
void     ProcInfo::ipAddr(int ip)       {_phy=ip;}
