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

  unsigned configure(const std::string& config_alias, XtcData::Xtc&, const void* bufEnd) override;
  unsigned beginrun(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& runInfo) override;
  size_t event(XtcData::Dgram&, const void* bufEnd, unsigned payloadSize) override;
  using Gpu::Detector::event;
public:
  void recordGraph(cudaStream_t&                      stream,
                   const unsigned&                    index,
                   const unsigned                     panel,
                   uint16_t const* const __restrict__ data) override;
private:
  enum {FexNamesIndex = NamesIndex::BASE};
private:
  std::vector<float*> m_peds_d;  // [NPanels][NGains][NPixels]
  std::vector<float*> m_gains_d; // [NPanels][NGains][NPixels]
};

  } // Gpu
} // Drp
