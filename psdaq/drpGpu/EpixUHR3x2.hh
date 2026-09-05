#pragma once

#include "Detector.hh"

#include "drp/EpixUHR3x2.hh"            // CPU-side Detector implementation
#include "drp/drp.hh"

namespace Drp {
  namespace Gpu {

class EpixUHR3x2 : public Gpu::Detector
{
public:
  EpixUHR3x2(Parameters& para, MemPoolGpu& pool);
  virtual ~EpixUHR3x2() override;

public:  // ePixUHR3x2 parameters:
  static const unsigned NumAsics    {     6 };
  static const unsigned NumRows     {   168 };  // elemRows    in drp/EpixUHR3x2.cc
  static const unsigned NumCols     {   192 };  // elemRowSize in drp/EpixUHR3x2.cc
  static const unsigned AsicPixels  { NumRows*NumCols };
  static const unsigned NPixels     { NumAsics*AsicPixels };
  // @todo: Confirm these against the firmware.  Taken from the ePixUHR values in
  //        EpixUHRemu/EpixUHRsim, i.e. gain-expanded 14-bit data with 2 gain
  //        bits above it.  drp/EpixUHR3x2.cc notes that 4 of the 16 bits are
  //        unused when the data is *not* gain-expanded, which would make these
  //        12 and 2 instead.
  static const unsigned RangeOffset {    14 };
  static const unsigned RangeBits   {     2 };
  static const unsigned NRanges     {     4 };

  // The payload is AxiStream Batcher formatted:
  //   tdest 0: Trigger (XPM), which is where the TimingHeader lives
  //   tdest 1: Event
  //   tdest 2: Timing
  //   tdest 3-8: ASIC data, in the order the two unbatchers deliver it
  static const unsigned NumSubFrames    { 9 };
  static const unsigned FirstDataTdest  { 3 };
  // The Reader concatenates the data sub-frames into the calibrated buffer in
  // tdest order, so the ASICs arrive scrambled.  Physical arrangement is
  //     A1 | A3 | A5
  //     A0 | A2 | A4
  // and the mapping is A0<->tdest6, A1<->tdest3, A2<->tdest7, A3<->tdest4,
  // A4<->tdest8, A5<->tdest5.  Anything writing XTC must descramble with this.
  static constexpr unsigned AsicForDataSubFrame[NumAsics] { 1, 3, 5, 0, 2, 4 };

public:
  unsigned configure(const std::string& config_alias, XtcData::Xtc&, const void* bufEnd) override;
  unsigned beginrun(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& runInfo) override;
  void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t count) override;
  using Gpu::Detector::event;
public:
  unsigned     rangeOffset()       const override { return RangeOffset; }
  unsigned     rangeBits()         const override { return RangeBits; }
  float const* pedestals_d()       const override { return m_pedsVec_d; };
  float const* gains_d()           const override { return m_gainsVec_d; };
  unsigned     subframeCount()     const override { return NumSubFrames; }
  unsigned     firstDataSubframe() const override { return FirstDataTdest; }
private:
  float* m_pedsVec_d;                   // [NRanges * NPixels]
  float* m_gainsVec_d;                  // [NRanges * NPixels]
};

  } // Gpu
} // Drp
