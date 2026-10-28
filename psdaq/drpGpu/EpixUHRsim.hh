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

public:  // ePixUHR3x2 parameters:
  static const unsigned NumRows    { 336 };
  static const unsigned NumCols    { 576 };
  static const unsigned NPixels    { NumRows*NumCols };
  static const unsigned RangeOffset{   0 };  // Gain in bit 0, an 11-bit value in bits 1-11
  static const unsigned RangeBits  {   1 };
  static const unsigned DataOffset {   1 };
  static const unsigned DataBits   {  11 };
  static const unsigned NRanges    { 1 << RangeBits };
public:
  unsigned configure  (const std::string& config_alias, XtcData::Xtc&, const void* bufEnd) override;
  unsigned beginrun   (XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info) override;
  void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t count) override;
  using Gpu::Detector::event;
public:
  unsigned     rangeOffset() const override { return RangeOffset; }
  unsigned     rangeBits()   const override { return RangeBits; }
  unsigned     dataOffset()  const override { return DataOffset; }
  unsigned     dataBits()    const override { return DataBits; }
  float const* pedestals_d() const override { return m_pedsVec_d; };
  float const* gains_d()     const override { return m_gainsVec_d; };

  // Launches the _event kernel template instantiated with PedGainCalib, from
  // this .so, so that the calibration is inlined into the kernel
  void recordEvent(cudaStream_t, unsigned blocks, unsigned threads,
                   const EventKernelArgs&) override;

  void         issuePhase2(XtcData::TransitionId::Value tid) override;
  float const* referenceBuffers() const override;
  unsigned     referenceBufCnt()  const override;
private:
  float* m_pedsVec_d;                   // [NRanges * NPixels]
  float* m_gainsVec_d;                  // [NRanges * NPixels]
};

  } // Gpu
} // Drp
