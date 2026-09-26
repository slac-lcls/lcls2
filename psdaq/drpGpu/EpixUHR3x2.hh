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

  // The payload is AxiStream Batcher formatted:
  //   tdest 0: Trigger (XPM), which is where the TimingHeader lives
  //   tdest 1: Event
  //   tdest 2: Timing
  //   tdest 3-8: ASIC data, one sub-frame per ASIC in ASIC order
  static const unsigned NumSubFrames    { 9 };
  static const unsigned FirstDataTdest  { 3 };
  // Data sub-frames are concatenated in tdest order, which is the order offline
  // expects: Gabriel confirms (2026-09-25) that Drp::EpixUHR3x2 writes its array
  // that way and that the order is correct, so tdest 3+k is ASIC k and no
  // remapping is needed.  An AsicForDataSubFrame table asserting otherwise used
  // to live here; it was never referenced and its premise was wrong.

public:
  unsigned configure(const std::string& config_alias, XtcData::Xtc&, const void* bufEnd) override;
  unsigned beginrun(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& runInfo) override;
  void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t count) override;
  using Gpu::Detector::event;
public:
  // The panel's data arrives already calibrated to fp16 by the detector's
  // firmware, so there is no pedestal or gain correction to do on the GPU and no
  // gain range encoded in the data: the per-element work is an fp16 -> fp32
  // conversion.  These four exist only to satisfy the base class.
  unsigned     rangeOffset()       const override { return 0;       /* Not used */ }
  unsigned     rangeBits()         const override { return 0;       /* Not used */ }
  float const* pedestals_d()       const override { return nullptr; /* Not used */ }
  float const* gains_d()           const override { return nullptr; /* Not used */ }
  unsigned     subframeCount()     const override { return NumSubFrames; }
  unsigned     firstDataSubframe() const override { return FirstDataTdest; }

  // Bytes of raw data to reserve ahead of each reduce buffer's payload.  Non-zero
  // only in pass-through mode, where the recorded data *is* the raw frame.  Fixed
  // size, with zeros for any ASIC that is withheld or delivers short, so offline
  // sees one shape whatever the ASIC configuration -- as the CPU DRP does.
  size_t       rawSize()           const override
  { return m_passthru ? NPixels * sizeof(uint16_t) : 0; }

  // [NumAsics][AsicPixels], the same shape Drp::EpixUHR3x2 writes, so that offline
  // sees one array whichever DRP produced the file
  unsigned     rawShape(unsigned* shape) const override
  {
    if (!m_passthru)  return 0;
    shape[0] = NumAsics;
    shape[1] = AsicPixels;
    return 2;
  }

  // Launches the _event kernel template instantiated with this detector's
  // per-element policy, from this .so, so that it is inlined into the kernel
  void recordEvent(cudaStream_t, unsigned blocks, unsigned threads,
                   const EventKernelArgs&) override;
private:
  // Record the panel's data as it arrives, uncalibrated and unreduced.  Set from
  // the `raw` kwarg for now; stage 2 will derive it from the CALIB config alias.
  bool m_passthru{false};
};

  } // Gpu
} // Drp
