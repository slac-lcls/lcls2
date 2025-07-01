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
  size_t event(XtcData::Dgram&, const void* bufEnd, unsigned payloadSize) override;
  using Gpu::Detector::event;
public:
  void recordGraph(cudaStream_t&                      stream,
                   const unsigned&                    index,
                   const unsigned                     panel,
                   uint16_t const* const __restrict__ data) override;
private:
  enum {FexNamesIndex = NamesIndex::BASE};
private:
  unsigned m_nPixels;
};

  } // Gpu
} // Drp
