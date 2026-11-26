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
  static const unsigned NumRows     {   336 };  // The panel's image, as the f/w lays it out
  static const unsigned NumCols     {   576 };
  static const unsigned NPixels     { NumRows*NumCols };
  // Gain ranges, i.e. 1 << rangeBits(): the single gain bit selects one of two
  // pedestal/gain planes
  static const unsigned NRanges     {     2 };

  // The payload is AxiStream Batcher formatted:
  //   tdest 0: Trigger (XPM), which is where the TimingHeader lives
  //   tdest 1: Event
  //   tdest 2: Timing
  //   tdest 3: The image, [NumRows][NumCols] row-major
  //   tdest 4: Header, 16 B per 3 ASICs: version, config status, maximum value
  static const unsigned NumSubFrames    { 5 };
  static const unsigned DataTdest       { 3 };

public:
  unsigned configure(const std::string& config_alias, XtcData::Xtc&, const void* bufEnd) override;
  unsigned beginrun(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& runInfo) override;
  void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t count) override;
  using Gpu::Detector::event;
public:
  // A u16 pixel: gain in bit 0, below an 11-bit value in bits 1-11.  Used only by the
  // u16 policy.
  unsigned     rangeOffset()       const override { return 0;  }  // The gain bit
  unsigned     rangeBits()         const override { return 1;  }
  unsigned     dataOffset()        const override { return 1;  }  // The ADC value
  unsigned     dataBits()          const override { return 11; }
  float const* pedestals_d()       const override { return m_peds_d;  }
  float const* gains_d()           const override { return m_gains_d; }
  unsigned     subframeCount()     const override { return NumSubFrames; }
  unsigned     firstDataSubframe() const override { return DataTdest; }

  // Capacity, so unconditional; fixed, with zeros for a short image, so offline sees
  // one shape
  size_t       rawSize()           const override
  { return NPixels * sizeof(uint16_t); }

  unsigned     rawShape(unsigned* shape) const override
  {
    shape[0] = NumRows;
    shape[1] = NumCols;
    return 2;
  }

  // Launches the _event kernel template instantiated with this detector's
  // per-element policy, from this .so, so that it is inlined into the kernel
  void recordEvent(cudaStream_t, unsigned blocks, unsigned threads,
                   const EventKernelArgs&) override;
private:
  // u16 data, calibrated on the GPU: the only `raw` kwarg value accepted (see the ctor)
  bool     m_u16{true};
  // One plane of pedestals and gains per gain range, laid out [NRanges][NPixels] as
  // pedGainCalibrate() indexes them.  Only allocated in u16 mode.
  float*   m_peds_d{nullptr};
  float*   m_gains_d{nullptr};
};

  } // Gpu
} // Drp
