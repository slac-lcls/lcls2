#pragma once

#include "drp/Detector.hh"

#include "MemPool.hh"                   // Needed for the base class
#include "psdaq/service/range.hh"
#include "psdaq/aes-stream-drivers/GpuAsyncUser.h"

#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace Drp {
  namespace Gpu {

// Defined in ReaderKernels.cuh, which only .cu files include: Detector.cc is
// compiled by the host compiler and must not see __global__ code
struct EventKernelArgs;

enum { MaxPnlsPerNode = 10 };       // From BEBDetector.hh
enum { ConfigNamesIndex = Drp::NamesIndex::BASE,
       EventNamesIndex  = unsigned(ConfigNamesIndex) + unsigned(MaxPnlsPerNode),
       FexNamesIndex    = unsigned(EventNamesIndex)  + unsigned(MaxPnlsPerNode),
       ReducerNamesIndex,
       ReducerConfigNamesIndex };         // index for xtc NamesId

class Detector : public Drp::Detector
{
public:
  Detector(Parameters* para, MemPoolGpu* pool) :
    Drp::Detector(para, pool),
    m_det(nullptr)
  {}
  virtual ~Detector() { if (m_det)  delete m_det; }

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

  unsigned configureScan(const nlohmann::json& scanKeys, XtcData::Xtc& xtc, const void* bufEnd) override;
  unsigned stepScan     (const nlohmann::json& stepInfo, XtcData::Xtc& xtc, const void* bufEnd) override;

  Pds::TimingHeader* getTimingHeader(uint32_t index) const override
  {
    auto       memPool    = m_pool->getAs<MemPoolGpu>();
    const auto dmaBuffers = memPool->hostWrtBufs();
    const auto cnt        = memPool->hostWrtBufsSize()/sizeof(*dmaBuffers);
    auto       dmaDsc     = (DmaDsc*)&dmaBuffers[index * cnt];
    return reinterpret_cast<Pds::TimingHeader*>(&dmaDsc[1]);
  }

  // The gain-range and data fields of a raw u16 pixel, whose order differs between
  // detectors.  The defaults put the range above the data (e.g. range 14/2, data 0/14).
  virtual unsigned     rangeOffset() const = 0;
  virtual unsigned     rangeBits()   const = 0;
  virtual unsigned     dataOffset()  const { return 0; }
  virtual unsigned     dataBits()    const { return rangeOffset(); }
  virtual float const* pedestals_d() const = 0;
  virtual float const* gains_d()     const = 0;

  // An L1Accept's highest tdest plus one, counting the leading timing sub-frames.
  // Zero: no batcher, and the payload is the TimingHeader then one contiguous block.
  virtual unsigned     subframeCount()     const { return 0; }
  // The tdest of the first sub-frame bearing detector data, when subframeCount() > 0
  virtual unsigned     firstDataSubframe() const { return 0; }

  // Record raw, uncalibrated and unreduced, as the CALIB config alias asks.  Call only
  // from Configure: buffer sizes, the reducer, the Reader graph and the Names entry
  // are all fixed there.  Cleared too, since the Detector outlives the cycle.
  virtual void         setPassthru(bool v)       { m_passthru = v; }
  bool                 passthru()          const { return m_passthru; }

  // The raw block's size, recorded with or instead of the reduced data; 0 for a
  // Detector with no raw mode.  CAPACITY, not per-event presence: it must not depend
  // on passthru(), since raw can arrive in any run.  Unused, it costs no disk.
  virtual size_t       rawSize()           const { return 0; }

  // Fills `shape` with the raw block's and returns the rank, or 0 for flat bytes
  virtual unsigned     rawShape(unsigned* shape) const { return 0; }

  // Describe this event's raw block in the Xtc, BEFORE the Reducer's description,
  // since raw precedes reduced in the buffer.  Virtual because Detector.cc is built
  // into each plugin, so drp_gpu can reach it only through the vtable.
  virtual void         rawEvent(XtcData::Xtc& xtc, const void* bufEnd);

  // Record this detector's instantiation of the _event kernel into the stream, once
  // per graph recording (see ReaderKernels.cuh)
  virtual void recordEvent(cudaStream_t    stream,
                           unsigned        blocks,
                           unsigned        threads,
                           const EventKernelArgs&) = 0;

  // Messages for the operator from the last transition, e.g. calibration defaults used
  std::vector<std::string> takeWarnings()  { return std::move(m_warnings); }
  // A short reason for the operator when the last transition failed, or empty
  std::string takeError()  { return std::move(m_error); }

  virtual void         issuePhase2(XtcData::TransitionId::Value) {} // Used in simulator mode only
  virtual float const* referenceBuffers() const { return nullptr; } // Used in simulator mode only
  virtual unsigned     referenceBufCnt()  const { return 0; }       // Used in simulator mode only
protected:
  template<typename T>
  void _initialize(Parameters& para, MemPoolGpu& pool) {
    // Create a Drp::Detector for the panel/PGP device
    m_det = new T(&para, &pool);
  }
protected:
  Drp::Detector* m_det;
  bool           m_passthru{false};     // See setPassthru()
  std::vector<std::string> m_warnings;  // See takeWarnings()
  std::string              m_error;     // See takeError()
};

  } // Gpu
} // Drp


extern "C"
{
  typedef Drp::Gpu::Detector* DetectorFactoryFn_t(Drp::Parameters&, Drp::Gpu::MemPoolGpu&);

  Drp::Gpu::Detector* createDetector(Drp::Parameters& para, Drp::Gpu::MemPoolGpu& pool);
}
