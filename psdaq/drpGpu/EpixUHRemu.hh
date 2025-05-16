#pragma once

#include "Detector.hh"

#include "drp/AreaDetector.hh"          // Detector implementation
#include "drp/drp.hh"

namespace Drp {
  namespace Gpu {

class EpixUHRemu : public Gpu::Detector
{
public:
  EpixUHRemu(Parameters& para, MemPoolGpu& pool);
  virtual ~EpixUHRemu() override;

public:  // ePixUHR parameters:
  static const unsigned NumAsics   {   6 };
  static const unsigned NumRows    { 192 };
  static const unsigned NumCols    { 168 };
  static const unsigned NPixels    { NumAsics*NumRows*NumCols };
  static const unsigned GainOffset {  14 };
  static const unsigned GainBits   {   2 };
  static const unsigned NGains     {   4 };
public:
  Gpu::Detector* gpuDetector() override { return this; }

  unsigned beginrun(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& runInfo) override;
public:
  void recordGraph(cudaStream_t&                      stream,
                   const unsigned&                    index,
                   const unsigned                     panel,
                   uint16_t const* const __restrict__ data) override;
  // @todo: To be implemented or moved
  //void recordReduceGraph(cudaStream_t&             stream,
  //                       const unsigned&           index,
  //                       float* const __restrict__ calibBuffers,
  //                       float* const __restrict__ dataBuffers) override;
private:
  std::vector<float*> m_peds_h;  // [NPanels][NGains][NPixels]
  std::vector<float*> m_gains_h; // [NPanels][NGains][NPixels]
};

  } // Gpu
} // Drp
