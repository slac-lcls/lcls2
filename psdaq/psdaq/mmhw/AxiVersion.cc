#include "psdaq/mmhw/AxiVersion.hh"

std::string Pds::Mmhw::AxiVersion::serialID() const
{
  uint32_t tmp[5];
  for(unsigned i=0; i<4; i++)
    tmp[i] = dnaValue[i];
  tmp[4] = 0;
  return std::string(reinterpret_cast<const char*>(tmp));
}

std::string Pds::Mmhw::AxiVersion::buildStamp() const
{
  uint32_t tmp[64];
  for(unsigned i=0; i<64; i++)
    tmp[i] = BuildStamp[i];
  return std::string(reinterpret_cast<const char*>(tmp));
}

