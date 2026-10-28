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

public:  // ePixUHR3x2 parameters, though the TDetSim payload is only a sequence:
  static const unsigned NumRows    { 336 };
  static const unsigned NumCols    { 576 };
  static const unsigned NPixels    { NumRows*NumCols };
  static const unsigned NRanges    {   2 };  // 1 << rangeBits()
public:
  unsigned configure(const std::string& config_alias, XtcData::Xtc&, const void* bufEnd) override;
  unsigned beginrun(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& runInfo) override;
  void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t count) override;
  using Gpu::Detector::event;
public:
  // As the ePixUHR3x2: gain in bit 0, an 11-bit value in bits 1-11
  unsigned     rangeOffset() const override { return 0;  }
  unsigned     rangeBits()   const override { return 1;  }
  unsigned     dataOffset()  const override { return 1;  }
  unsigned     dataBits()    const override { return 11; }
  float const* pedestals_d() const override { return m_pedsVec_d; };
  float const* gains_d()     const override { return m_gainsVec_d; };

  // Bytes of raw data the detector produces, which sizes whichever buffer holds it:
  // every reduce buffer in CALIB, where raw is the whole payload, or a prescale slot
  // when prescaling in BEAM.  Unconditional, because this is capacity -- whether a
  // given event uses it is the keepRaw bit, unknown at Configure.  Fixed size, with
  // zeros for a short payload, so offline sees one shape whatever arrives.
  size_t       rawSize()     const override
  { return NPixels * sizeof(uint16_t); }     // u16 per pixel, as rawShape() declares

  // The panel's image shape, as Gpu::EpixUHR3x2 writes it
  unsigned     rawShape(unsigned* shape) const override
  {
    shape[0] = NumRows;
    shape[1] = NumCols;
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
