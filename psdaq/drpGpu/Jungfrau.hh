#pragma once

#include "Detector.hh"

#include "drp/Jungfrau.hh"              // CPU-side Detector implementation
#include "drp/JungfrauData.hh"          // Wire format: packets, headers, geometry
#include "drp/drp.hh"

namespace Drp {
  namespace Gpu {

// The Jungfrau's payload is AxiStream Batcher formatted, and unlike the
// EpixUHR3x2's it is batched twice:
//
//   tdest 0        : the TimingHeader
//   tdest 1        : auxiliary timing
//   tdest 2        : module 0's data
//   tdest 3        : unused
//   tdest 4, 5, ... : modules 1, 2, ...        (see Drp::Jungfrau::_event)
//
// so the data tdests are not a contiguous run, and each module's sub-frame is
// itself a batch of JungfrauData::PacketNum UDP packets.  Each packet carries a
// JungfrauData::Header followed by PixelPerPacket uint16 pixels, and the header's
// packetnum -- not the packet's position in the batch -- says where those pixels
// belong in the frame.
class Jungfrau : public Gpu::Detector
{
public:
  Jungfrau(Parameters& para, MemPoolGpu& pool);
  virtual ~Jungfrau() override;

public:  // Jungfrau parameters, from drp/JungfrauData.hh
  static const unsigned Rows           { JungfrauData::Rows };           // 512
  static const unsigned Cols           { JungfrauData::Cols };           // 1024
  static const unsigned ModulePixels   { JungfrauData::PixelNum };       // 512*1024
  static const unsigned PacketNum      { JungfrauData::PacketNum };      // 128
  static const unsigned PixelPerPacket { JungfrauData::PixelPerPacket }; // 4096
  // Gain is encoded in the top 2 bits of each 16-bit pixel; see the gain_bits
  // and data_bits masks in Drp::Jungfrau::_countNumHotPixels()
  static const unsigned RangeOffset    {  14 };
  static const unsigned RangeBits      {   2 };
  static const unsigned NRanges        {   4 };

  // The tdest bearing module 0's data.  Later modules are at FirstDataTdest + 2
  // onwards, skipping tdest 3, hence the mapping below rather than a plain run.
  static const unsigned FirstDataTdest {   2 };

  __host__ __device__
  static unsigned tdestOfModule(unsigned module)
  { return module == 0 ? FirstDataTdest : module + 3; }

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
  // The highest tdest in use, plus one
  unsigned     subframeCount()     const override
  { return tdestOfModule(m_nModules - 1) + 1; }
  unsigned     firstDataSubframe() const override { return FirstDataTdest; }

  void recordEvent(cudaStream_t, unsigned blocks, unsigned threads,
                   const EventKernelArgs&) override;

  auto nModules() const { return m_nModules; }
  auto nPixels()  const { return m_nModules * ModulePixels; }
private:
  unsigned m_nModules;                  // From the lane mask, as on the CPU side
  float*   m_pedsVec_d;                 // [NRanges * nPixels()]
  float*   m_gainsVec_d;                // [NRanges * nPixels()]
};

  } // Gpu
} // Drp
