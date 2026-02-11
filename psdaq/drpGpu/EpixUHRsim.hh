#pragma once

#include "Detector.hh"

#include "xtcdata/xtc/TransitionId.hh"
#include "drp/AreaDetector.hh"          // Detector implementation
#include "drp/drp.hh"

namespace Drp {
  namespace Gpu {

class EpixUHRsim : public Gpu::Detector
{
public:
  EpixUHRsim(Parameters& para, MemPoolGpu& pool);
  virtual ~EpixUHRsim() override;

public:  // ePixUHR parameters:
  static const unsigned NumAsics   {   6 };
  static const unsigned NumRows    { 192 };
  static const unsigned NumCols    { 168 };
  static const unsigned NPixels    { NumAsics*NumRows*NumCols };
  static const unsigned RangeOffset{  14 };
  static const unsigned RangeBits  {   2 };
  static const unsigned NRanges    {   4 };
public:
  unsigned configure  (const std::string& config_alias, XtcData::Xtc&, const void* bufEnd) override;
  unsigned beginrun   (XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info) override;
  void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t count) override;
  using Gpu::Detector::event;
public:
//  __device__ void calibrate(float*    const calib,
//                            uint16_t* const raw,
//                            unsigned  const count,
//                            unsigned  const nPanels) const;
  unsigned rangeOffset() const override { return RangeOffset; }
  unsigned rangeBits()   const override { return RangeBits; }

  void recordGraph(cudaStream_t          stream,
                   const unsigned&       index,
                   const unsigned        panel,
                   uint16_t const* const data) override;

  void issuePhase2(XtcData::TransitionId::Value) override;
private:
  std::vector<float*> m_pedsVec_d;  // [nPanels][NRanges * NPixels]
  std::vector<float*> m_gainsVec_d; // [nPanels][NRanges * NPixels]
//  float** m_pedArr_d;               // [nPanels][NRanges * NPixels]
//  float** m_gainArr_d;              // [nPanels][NRanges * NPixels]
};

  } // Gpu
} // Drp
