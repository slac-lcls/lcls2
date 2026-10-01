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
  // Gain ranges, i.e. 1 << rangeBits(): the single gain bit selects one of two
  // pedestal/gain planes
  static const unsigned NRanges     {     2 };

  // The payload is AxiStream Batcher formatted:
  //   tdest 0: Trigger (XPM), which is where the TimingHeader lives
  //   tdest 1: Event
  //   tdest 2: Timing
  //   tdest 3-8: ASIC data, one sub-frame per ASIC in ASIC order
  static const unsigned NumSubFrames    { 9 };
  static const unsigned FirstDataTdest  { 3 };
  // Data sub-frames are concatenated in tdest order, which is the order offline
  // expects: tdest 3+k is ASIC k and no remapping is needed (Gabriel, 2026-09-25,
  // confirming that Drp::EpixUHR3x2 writes its array that way).  The physical
  // arrangement invites the opposite conclusion, so resist adding a remap here
  // without checking what offline actually reads.

public:
  unsigned configure(const std::string& config_alias, XtcData::Xtc&, const void* bufEnd) override;
  unsigned beginrun(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& runInfo) override;
  void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t count) override;
  using Gpu::Detector::event;
public:
  // Where the gain bit and the ADC value sit in a u16 pixel: gain in bit 0, an 11-bit
  // value in bits 1-11, zeros in bits 12-15 (Gabriel, 2026-09-28).  The gain bit is
  // *below* the data, unlike some other detectors, which is why Gpu::Detector locates
  // both fields explicitly instead of deriving the data field's width from
  // rangeOffset().
  //
  // Consulted only by the u16 policy: an fp16 payload is calibrated by the firmware,
  // so its policy converts width and ignores these.
  unsigned     rangeOffset()       const override { return 0;  }  // The gain bit
  unsigned     rangeBits()         const override { return 1;  }
  unsigned     dataOffset()        const override { return 1;  }  // The ADC value
  unsigned     dataBits()          const override { return 11; }
  float const* pedestals_d()       const override { return m_peds_d;  }
  float const* gains_d()           const override { return m_gains_d; }
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
  // Nb: m_passthru, which gates rawSize() and rawShape() above, is the base class's:
  // the CALIB config alias sets it per Configure.  See Gpu::Detector::setPassthru().
  //
  // The panel's data is u16 rather than fp16, so the GPU applies pedestals and gains.
  // Selected by `raw=u16`; `raw=fp16` is the default.
  bool     m_u16{false};
  // One plane of pedestals and gains per gain range, laid out [NRanges][NPixels] as
  // pedGainCalibrate() indexes them.  Only allocated in u16 mode.
  float*   m_peds_d{nullptr};
  float*   m_gains_d{nullptr};
};

  } // Gpu
} // Drp
