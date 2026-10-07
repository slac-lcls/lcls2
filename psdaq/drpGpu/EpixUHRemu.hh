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
  static const unsigned NumRows    { 168 };  // elemRows    in drp/EpixUHR3x2.cc
  static const unsigned NumCols    { 192 };  // elemRowSize in drp/EpixUHR3x2.cc
  static const unsigned AsicPixels { NumRows*NumCols };
  static const unsigned NPixels    { NumAsics*AsicPixels };
  static const unsigned RangeOffset{  14 };
  static const unsigned RangeBits  {   2 };
  static const unsigned NRanges    {   4 };
public:
  unsigned configure(const std::string& config_alias, XtcData::Xtc&, const void* bufEnd) override;
  unsigned beginrun(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& runInfo) override;
  void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t count) override;
  using Gpu::Detector::event;
public:
  unsigned     rangeOffset() const override { return RangeOffset; }
  unsigned     rangeBits()   const override { return RangeBits; }
  float const* pedestals_d() const override { return m_pedsVec_d; };
  float const* gains_d()     const override { return m_gainsVec_d; };

  // Bytes of raw data the detector produces, which sizes whichever buffer holds it:
  // every reduce buffer in CALIB, where raw is the whole payload, or a prescale slot
  // when prescaling in BEAM.  Unconditional, because this is capacity -- whether a
  // given event uses it is the keepRaw bit, unknown at Configure.  Fixed size, with
  // zeros for a short payload, so offline sees one shape whatever arrives.
  size_t       rawSize()     const override
  { return NPixels * sizeof(uint16_t); }     // u16 per pixel, as rawShape() declares

  // [NumAsics][AsicPixels], the same shape Drp::EpixUHR3x2 and Gpu::EpixUHR3x2 write,
  // so that offline sees one array whichever detector or DRP produced the file.  The
  // emulator's payload arrives as one contiguous run of u16 rather than the real
  // detector's per-ASIC sub-frames, but it arrives in ASIC order, so this describes
  // the bytes where they already are and needs no remapping.
  unsigned     rawShape(unsigned* shape) const override
  {
    shape[0] = NumAsics;
    shape[1] = AsicPixels;
    return 2;
  }

  // Launches the _event kernel template instantiated with PedGainCalib, from
  // this .so, so that the calibration is inlined into the kernel
  void recordEvent(cudaStream_t, unsigned blocks, unsigned threads,
                   const EventKernelArgs&) override;
private:
  float* m_pedsVec_d;                   // [NRanges * NPixels]
  float* m_gainsVec_d;                  // [NRanges * NPixels]
};

  } // Gpu
} // Drp
