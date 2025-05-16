#pragma once

#include "AreaDetector.hh"              // Detector implementation
//#include "Detector.hh"

#include "drp.hh"

namespace Drp {

class Parameters;
class MemPool;

  //class AreaDetectorGpu : public Detector
class AreaDetectorGpu : public AreaDetector
{
public:
  AreaDetectorGpu(Parameters& para, MemPool& pool);
  virtual ~AreaDetectorGpu() {}
//public:
//  unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd) override;
//  unsigned beginrun(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& runInfo) override;
//  void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event) override;
public:
  // __device__ void event(const TimingHeader&, PGPEvent*);
  // __device__ void slowUpdate(const TimingHeader&);
private:
  std::vector<Parameters>   m_paras;
  //std::vector<AreaDetector> m_dets;
};

}
