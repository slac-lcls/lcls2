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

  // Launches the _event kernel template instantiated with this detector's
  // fp16 -> fp32 policy, from this .so, so that it is inlined into the kernel
  void recordEvent(cudaStream_t, unsigned blocks, unsigned threads,
                   const EventKernelArgs&) override;
};

  } // Gpu
} // Drp
