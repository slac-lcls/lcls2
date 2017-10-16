#include "psdaq/cphw/AxiVersion.hh"

using namespace Pds::Cphw;

std::string AxiVersion::buildStamp() const
{
  uint32_t buff[64];
  for(unsigned i=0; i<64; i++) {
    buff[i]= BuildStamp[i];
  }
  return std::string(reinterpret_cast<const char*>(buff));
}
