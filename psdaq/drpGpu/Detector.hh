#pragma once

#include "drp/Detector.hh"

#include "MemPool.hh"                   // Needed for the base class
#include "psdaq/service/range.hh"

#include <cuda_runtime.h>

namespace Drp {
  namespace Gpu {

class Detector : public Drp::Detector
{
public:
  Detector(Parameters* para, MemPoolGpu* pool) : Drp::Detector(para, pool) {}
  virtual ~Detector() { printf("*** Gpu::Detector: dtor\n"); for (const auto& det : m_dets) { delete det; } }

  Gpu::Detector* gpuDetector() override { return this; }

  nlohmann::json connectionInfo(const nlohmann::json& msg) override;
  void connectionShutdown() override;
  void connect(const nlohmann::json& connect_json, const std::string& collectionId) override;
  unsigned configure(const std::string& config_alias, XtcData::Xtc& xtc, const void* bufEnd) override;
  unsigned beginrun (XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& runInfo) override;
  unsigned beginstep(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& stepInfo) override;
  unsigned enable   (XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info) override;
  unsigned disable  (XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info) override;
  void slowupdate(XtcData::Xtc& xtc, const void* bufEnd) override { /* Not used */ };
  void event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent* event) override { /* Not used */ }
  using Drp::Detector::event;
  void shutdown() override;

  // @todo: What to do about these?
  //// Scan methods.  Default is to fail.
  //unsigned configureScan(const nlohmann::json& stepInfo, XtcData::Xtc& xtc, const void* bufEnd) override {return 1;};
  //unsigned stepScan     (const nlohmann::json& stepInfo, XtcData::Xtc& xtc, const void* bufEnd) override {return 1;};

  Pds::TimingHeader* getTimingHeader(uint32_t index) const override
  {
    const auto& dmaBuffers = m_pool->getAs<MemPoolGpu>()->hostBuffers_h()[0]; // Reference only worker 0's
    auto dsc = dmaBuffers[index];
    static const unsigned DmaDscWords = sizeof(DmaDsc) / sizeof(uint32_t);
    return reinterpret_cast<Pds::TimingHeader*>(&dsc[DmaDscWords]);
  }

  virtual void recordGraph(cudaStream_t&                      stream,
                           const unsigned&                    index,
                           const unsigned                     panel,
                           uint16_t const* const __restrict__ data) = 0;
protected:
  template<typename T>
  void _initialize(Parameters& para, MemPoolGpu& pool) {
    // Create a copy of the Parameters and replace the PGP device name with the
    // real one for each Detector instance the GPU will service
    // @todo: This seems wasteful - is there a nicer solution?
    unsigned i = 0;
    std::vector<int> panels;
    auto pos = para.device.find("_", 0);
    Pds::getRange(para.device.substr(pos+1, para.device.length()), panels);
    for (const auto& unit : panels) {
      m_params.push_back(para);
      m_params[i].device = para.device.substr(0, pos+1) + std::to_string(unit);

      // Create a Drp::Detector for each panel/PGP device
      m_dets.push_back(new T(&m_params[i], &pool));

      ++i;
    }
  }
protected:
  std::vector<Parameters>     m_params;
  std::vector<Drp::Detector*> m_dets;
};

  } // Gpu
} // Drp


extern "C"
{
  typedef Drp::Gpu::Detector* DetectorFactoryFn_t(Drp::Parameters&, Drp::Gpu::MemPoolGpu&);

  Drp::Gpu::Detector* createDetector(Drp::Parameters& para, Drp::Gpu::MemPoolGpu& pool);
}
