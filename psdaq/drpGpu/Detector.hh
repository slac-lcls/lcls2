#pragma once

#include "drp/Detector.hh"

#include "MemPool.hh"                   // Needed for the base class
#include "psdaq/service/range.hh"

#include <cuda_runtime.h>

namespace Drp {
  namespace Gpu {

enum { MaxPnlsPerNode = 10 };       // From BEBDetector.hh
enum { ConfigNamesIndex = Drp::NamesIndex::BASE,
       EventNamesIndex  = unsigned(ConfigNamesIndex) + unsigned(MaxPnlsPerNode),
       FexNamesIndex    = unsigned(EventNamesIndex)  + unsigned(MaxPnlsPerNode),
       ReducerNamesIndex };         // index for xtc NamesId

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
  using Drp::Detector::event;
  void shutdown() override;

  // @todo: What to do about these?
  //// Scan methods.  Default is to fail.
  //unsigned configureScan(const nlohmann::json& stepInfo, XtcData::Xtc& xtc, const void* bufEnd) override {return 1;};
  //unsigned stepScan     (const nlohmann::json& stepInfo, XtcData::Xtc& xtc, const void* bufEnd) override {return 1;};

  Pds::TimingHeader* getTimingHeader(uint32_t index) const override
  {
    auto       memPool    = m_pool->getAs<MemPoolGpu>();
    const auto dmaBuffers = memPool->hostWrtBufsVec_h()[0]; // Reference only panel 0's
    const auto cnt        = memPool->hostWrtBufsSize()/sizeof(*dmaBuffers);
    auto       dsc        = &dmaBuffers[index * cnt];
    constexpr unsigned DmaDscWords = sizeof(DmaDsc) / sizeof(uint32_t);
    return reinterpret_cast<Pds::TimingHeader*>(&dsc[DmaDscWords]);
  }

  //// Device methods can't be virtual due to the vtable not containing device pointers
  //__device__ void calibrate(float*    const calib,
  //                          uint16_t* const raw,
  //                          unsigned  const count,
  //                          unsigned  const nPanels,
  //                          unsigned  const rangeOffset,
  //                          unsigned  const rangeBits) const;
  virtual unsigned rangeOffset() const = 0;
  virtual unsigned rangeBits()   const = 0;

  virtual void recordGraph(cudaStream_t&         stream,
                           const unsigned&       index_d,
                           const unsigned        panel,
                           uint16_t const* const data) = 0;

  virtual void issuePhase2(XtcData::TransitionId::Value) {} // Used in simulator mode only
protected:
  template<typename T>
  void _initialize(Parameters& para, MemPoolGpu& pool) {
    // Create a copy of the Parameters and replace the PGP device name with the
    // real one for each Detector instance the GPU will service
    // @todo: This seems wasteful - is there a nicer solution?
    unsigned i = 0;
    for (const auto& panel : pool.panels()) {
      m_params.push_back(para);
      m_params[i].device = panel.name;

      // Create a Drp::Detector for each panel/PGP device
      m_dets.push_back(new T(&m_params[i], &pool));

      ++i;
    }
  }
protected:
  std::vector<Parameters>     m_params;
  std::vector<Drp::Detector*> m_dets;
public: // @todo: fix this
  float**                     m_pedArr_d;  // [nPanels][NRanges * NPixels]
  float**                     m_gainArr_d; // [nPanels][NRanges * NPixels]
};

  } // Gpu
} // Drp


extern "C"
{
  typedef Drp::Gpu::Detector* DetectorFactoryFn_t(Drp::Parameters&, Drp::Gpu::MemPoolGpu&);

  Drp::Gpu::Detector* createDetector(Drp::Parameters& para, Drp::Gpu::MemPoolGpu& pool);
}
