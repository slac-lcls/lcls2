#pragma once

#include "drp/Detector.hh"

#include "MemPool.hh"                   // Needed for the base class
#include "psdaq/service/range.hh"
#include "psdaq/aes-stream-drivers/GpuAsyncUser.h"

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
  Detector(Parameters* para, MemPoolGpu* pool) :
    Drp::Detector(para, pool),
    m_det(nullptr)
  { printf("*** Gpu::Detector::ctor: this %p, &det %p\n", this, &m_det); }
  virtual ~Detector() { printf("*** Gpu::Detector: dtor\n"); if (m_det)  delete m_det; }

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
    const auto dmaBuffers = memPool->hostWrtBufs_h();
    const auto cnt        = memPool->hostWrtBufsSize()/sizeof(*dmaBuffers);
    auto       dmaDsc     = (DmaDsc*)&dmaBuffers[index * cnt];
    return reinterpret_cast<Pds::TimingHeader*>(&dmaDsc[1]);
  }

  //// Device methods can't be virtual due to the vtable not containing device pointers
  //__device__ void calibrate(float*    const calib,
  //                          uint16_t* const raw,
  //                          unsigned  const count,
  //                          unsigned  const rangeOffset,
  //                          unsigned  const rangeBits) const;
  virtual unsigned     rangeOffset() const = 0;
  virtual unsigned     rangeBits()   const = 0;
  virtual float const* pedestals_d() const = 0;
  virtual float const* gains_d()     const = 0;

  virtual void recordGraph(cudaStream_t          stream,
                           const unsigned&       index_d,
                           uint16_t const* const data) = 0;

  virtual void issuePhase2(XtcData::TransitionId::Value) {} // Used in simulator mode only
  virtual float const* referenceBuffers() const { return nullptr; } // Used in simulator mode only
  virtual unsigned     referenceBufCnt()  const { return 0; }       // Used in simulator mode only
protected:
  template<typename T>
  void _initialize(Parameters& para, MemPoolGpu& pool) {
    // Create a Drp::Detector for the panel/PGP device
    m_det = new T(&para, &pool);
    printf("*** Gpu::Detector:_init: this %p, &det %p, det %p\n", this, &m_det, m_det);
  }
protected:
  Drp::Detector* m_det;
};

  } // Gpu
} // Drp


extern "C"
{
  typedef Drp::Gpu::Detector* DetectorFactoryFn_t(Drp::Parameters&, Drp::Gpu::MemPoolGpu&);

  Drp::Gpu::Detector* createDetector(Drp::Parameters& para, Drp::Gpu::MemPoolGpu& pool);
}
