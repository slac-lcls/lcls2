#pragma once

#include "Detector.hh"

#include "drp/AreaDetector.hh"          // Detector implementation
#include "drp/drp.hh"

namespace Drp {
  namespace Gpu {

class AreaDetector : public Gpu::Detector
{
public:
  AreaDetector(Parameters& para, MemPoolGpu& pool);
  virtual ~AreaDetector() override;
public:
  Gpu::Detector* gpuDetector() override { return this; }
public:
  unsigned configure(const std::string& config_alias, XtcData::Xtc&, const void* bufEnd) override;
  void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t count) override;
  using Gpu::Detector::event;
public:
  void recordGraph(cudaStream_t&                      stream,
                   const unsigned&                    index,
                   const unsigned                     panel,
                   uint16_t const* const __restrict__ data) override;
private:
  // @todo: Revisit: Must match detector definition
  enum { MaxPnlsPerNode = 10 };       // From BEBDetector.hh
  enum { ConfigNamesIndex = Drp::NamesIndex::BASE,
         EventNamesIndex  = unsigned(ConfigNamesIndex) + unsigned(MaxPnlsPerNode),
         FexNamesIndex    = unsigned(EventNamesIndex)  + unsigned(MaxPnlsPerNode),
         ReducerNamesIndex };         // index for xtc NamesId
private:
  unsigned m_nPixels;
};

  } // Gpu
} // Drp
