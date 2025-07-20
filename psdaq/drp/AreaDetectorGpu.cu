#include "AreaDetectorGpu.hh"
#include "psdaq/service/EbDgram.hh"

using namespace Drp;
using namespace XtcData;
using json = nlohmann::json;

AreaDetectorGpu::AreaDetectorGpu(Parameters& para, MemPool& pool) :
  //Detector(&para, &pool)
  AreaDetector(&para, &pool)
{
  // Create a copy of the Parameters and replace the PGP device name with the
  // real one for each Detector instance the GPU will service
  //// @todo: This seems wasteful - is there another solution?
  //unsigned i = 0;
  //std::vector<int> units;
  //auto pos = para.device.find("_", 0);
  //getRange(para.device.substr(pos+1, para.device.length()), units);
  //for (auto unit : units) {
  //  m_params.push_back(para);
  //  m_params[i].device = para.device + std:to_string(unit);
  //
  //  m_dets.push_back( {m_params[i], &pool} );
  //  ++i;
  //}
}

//unsigned AreaDetectorGpu::configure(const std::string& config_alias, Xtc& xtc, const void* bufEnd)
//{
//  // @todo: Fill in later
//    return 0;
//}
//
//unsigned AreaDetectorGpu::beginrun(Xtc& xtc, const void* bufEnd, const json& runInfo)
//{
//  // @todo: Fill in later
//    return 0;
//}
//
//void AreaDetectorGpu::event(Dgram& dgram, const void* bufEnd, PGPEvent* event)
//{
//  // @todo: Fill in later
//}

//__device__ void AreaDetectorGpu::event(const TimingHeader&, PGPEvent*) {}
//__device__ void AreaDetectorGpu::slowUpdate(const TimingHeader&) {}

// The class factory

extern "C" Detector* createDetectorGpu(Parameters& para, MemPool& pool)
{
  return new AreaDetectorGpu(para, pool);
}
