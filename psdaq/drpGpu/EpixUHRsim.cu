#include "EpixUHRsim.hh"

#include "ReaderKernels.cuh"

#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "psalg/utils/SysLog.hh"
#include "SimDetector.hh"

#include <vector>
#include <cstdlib>

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Pds;
using namespace Drp::Gpu;
using json = nlohmann::json;
using f32vec_t = std::vector<float>;
using u16vec_t = std::vector<uint16_t>;

struct uhr_domain{ static constexpr char const* name{"EpixUHRsim"}; };
using uhr_scoped_range = nvtx3::scoped_range_in<uhr_domain>;


namespace Drp {
  class PGPEvent;
  namespace Gpu {

// The functionality of the Drp::Detector is needed to set up the panel
// However, the data description and handling in the GPU case will be different,
// so we create a Drp Detector class to handle just the portion we need.
// Derive from Drp::SimDetector so it can be made non-abstract.
class SimDet : public SimDetector
{
public:
  SimDet(Parameters* para, MemPoolGpu* pool, unsigned len=100) : SimDetector(para, pool, len) {}
  using SimDetector::event;
  void event(Dgram&, const void* bufEnd, PGPEvent*, uint64_t count) override { /* Not used */ }

  unsigned     rangeOffset() const override { return 0; /* Not used */ }
  unsigned     rangeBits()   const override { return 0; /* Not used */ }
  float const* pedestals_d() const override { return nullptr; /* Not used */ }
  float const* gains_d()     const override { return nullptr; /* Not used */ }

  // This class only sets the panel up; EpixUHRsim launches the event kernel
  void recordEvent(cudaStream_t, unsigned, unsigned,
                   const EventKernelArgs&) override { /* Not used */ }

  void generateEvents(unsigned nEvents)
  {
    // Generate data for nEvents events
    m_nEvents = nEvents;
    m_raw.resize(nEvents * EpixUHRsim::NPixels);
    std::vector<uint64_t> nGainEvts(EpixUHRsim::NRanges, 0);
    for (unsigned i = 0; i < nEvents; ++i) {
      auto raw = m_raw.data() + i * EpixUHRsim::NPixels;

      for (unsigned j = 0; j < EpixUHRsim::NPixels; ++j) {
        auto datum = uint16_t((float(rand()) / float(RAND_MAX)) * (1 << EpixUHRsim::RangeOffset));
        auto range = (float(rand()) / float(RAND_MAX));
        unsigned gain;
        if      (range <= 0.25)  gain = 0;
        else if (range <= 0.5)   gain = 1;
        else if (range <= 0.75)  gain = 2;
        else if (range <= 1.0)   gain = 3;
        else {
          printf("*** Bad range value %f\n", range);
          gain = 3;
        }
        raw[j] = (gain << EpixUHRsim::RangeOffset) | datum;
        ++nGainEvts[gain];
        //if (j < 4) {
        //  printf("i %u, j %u: raw %04x, dat %u, rng %u\n", i, j, raw[j], datum, gain);
        //}
      }
    }
    printf("*** Number of pixels generated for each of %zu gain ranges:\n", nGainEvts.size());
    uint64_t total = 0;
    for (unsigned i = 0; i < nGainEvts.size(); ++i) {
      printf("  %9lu", nGainEvts[i]);
      total += nGainEvts[i];
    }
    printf("  total %lu\n", total);
    double sum = 0.0;
    for (unsigned i = 0; i < nGainEvts.size(); ++i) {
      printf("  %9f", double(nGainEvts[i]) / double(total));
      sum += double(nGainEvts[i]) / double(total);
    }
    printf("  total %f\n\n", sum);

    // Copy the raw event data to the device
    chkError(cudaMalloc(&m_raw_d,               m_raw.size() * sizeof(*m_raw_d)));
    chkError(cudaMemcpy( m_raw_d, m_raw.data(), m_raw.size() * sizeof(*m_raw_d), cudaMemcpyDefault));

    // Create the reference buffers here, but fill them in during BeginRun
    chkError(cudaMalloc(&m_reference_d, nEvents * EpixUHRsim::NPixels * sizeof(*m_reference_d)));
  }
  void calculateReference(const std::vector<f32vec_t>& pedestals,
                          const std::vector<f32vec_t>& gains)
  {
    // Generate calibrated events for reference
    std::vector<float> reference(m_nEvents * EpixUHRsim::NPixels);
    for (unsigned i = 0; i < m_nEvents; ++i) {
      auto raw = m_raw.data() + i * EpixUHRsim::NPixels;
      auto ref = reference.data() + i * EpixUHRsim::NPixels;

      for (unsigned j = 0; j < EpixUHRsim::NPixels; ++j) {
        auto gain  = (raw[j] >> EpixUHRsim::RangeOffset) & ((1 << EpixUHRsim::RangeBits) - 1);
        auto datum = raw[j] & ((1 << EpixUHRsim::RangeOffset) - 1);
        ref[j] = (float(datum) - pedestals[gain][j]) * gains[gain][j];
        //if (j < 4) {
        //  printf("***SimDet::calcRef: i %u, j %u: raw %04x, dat %u, rng %u, ped %f, gn %f, ref %f\n",
        //         i, j, raw[j], datum, gain, pedestals[gain][j], gains[gain][j], ref[j]);
        //}
      }
    }

    // Copy the reference data for all reference events to the device
    chkError(cudaMemcpy( m_reference_d, reference.data(), reference.size() * sizeof(*m_reference_d), cudaMemcpyDefault));
  }
  float const* referenceBuffers() const
  {
    return m_reference_d;
  }
  unsigned referenceBufCnt() const
  {
    return m_nEvents;
  }
protected:
  size_t _genL1Payload(uint8_t** buffer, size_t index, size_t bufSize) override
  {
    auto rawSize = m_raw.size() / m_nEvents;
    *buffer = (uint8_t*)(m_raw_d + (index % m_nEvents) * rawSize);
    rawSize *= sizeof(*m_raw_d);
    return ((bufSize > 0) && (bufSize < rawSize)) ? bufSize : rawSize;
  }
private:
  unsigned              m_nEvents;
  std::vector<uint16_t> m_raw;
  uint16_t*             m_raw_d;
  float*                m_reference_d;
};

class RawDef : public VarDef
{
public:
  enum index
    {
      array_raw
    };

  RawDef()
  {
    Alg raw("raw", 0, 0, 0);
    NameVec.push_back({"array_raw", Name::UINT8, 1});
  }
};
  } // Gpu
} // Drp


EpixUHRsim::EpixUHRsim(Drp::Parameters& para, MemPoolGpu& pool) :
  Gpu::Detector(&para, &pool)
{
  _initialize<Drp::Gpu::SimDet>(para, pool);

  // Check there is enough space in the DMA buffers for this many pixels
  assert(NPixels <= (pool.dmaSize() - sizeof(DmaDsc) - sizeof(TimingHeader)) / sizeof(uint16_t));

  // Set up buffers
  pool.createCalibBuffers(NPixels);

  // Allocate space for the calibration constants
  chkError(cudaMalloc(&m_pedsVec_d,  NRanges * NPixels * sizeof(*m_pedsVec_d)));
  chkError(cudaMalloc(&m_gainsVec_d, NRanges * NPixels * sizeof(*m_gainsVec_d)));

  // Generate fake raw event data and load it onto the device
  auto& simDet = *static_cast<SimDet*>(m_det);
  auto  nEvents{10};                    // @todo: TBR
  simDet.generateEvents(nEvents);
}

EpixUHRsim::~EpixUHRsim()
{
  auto pool = m_pool->getAs<MemPoolGpu>();
  if (m_gainsVec_d)  chkError(cudaFree(m_gainsVec_d));
  if (m_pedsVec_d)   chkError(cudaFree(m_pedsVec_d));

  pool->destroyCalibBuffers();
}

unsigned EpixUHRsim::configure(const std::string& config_alias, Xtc& xtc, const void* bufEnd)
{
  logging::info("Gpu::EpixUHRsim configure: this %p, &det %p, det %p", this, &m_det, m_det);

  unsigned rc = m_det->configure(config_alias, xtc, bufEnd);
  if (rc) {
    logging::error("Gpu::EpixUHRsim::configure failed for %s\n", m_para->device);
  }

  Alg alg("raw", 0, 0, 0);
  NamesId namesId(nodeId, EventNamesIndex);
  Names& names = *new(xtc, bufEnd) Names(bufEnd,
                                         m_para->detName.c_str(), alg,
                                         m_para->detType.c_str(), m_para->serNo.c_str(), namesId, m_para->detSegment);
  RawDef dataDef;
  names.add(xtc, bufEnd, dataDef);
  m_namesLookup[namesId] = NameIndex(names);

  logging::info("Gpu::EpixUHRsim configure: xtc size %u", xtc.sizeofPayload());

  return rc;
}

unsigned EpixUHRsim::beginrun(XtcData::Xtc& xtc, const void* bufEnd, const nlohmann::json& info)
{
  unsigned rc = m_det->beginrun(xtc, bufEnd, info);
  if (rc) {
    logging::error("Gpu::EpixUHRsim::beginrun failed for %s\n", m_para->device);
  }

  // Load the calibration constants onto the GPU
  std::vector<f32vec_t> peds(NRanges);
  std::vector<f32vec_t> gains(NRanges);
  auto pool = m_pool->getAs<MemPoolGpu>();
  auto peds_d  = m_pedsVec_d;
  auto gains_d = m_gainsVec_d;
  for (unsigned range = 0; range < NRanges; ++range) {
    peds[range].resize(NPixels);
    gains[range].resize(NPixels);
    for (unsigned pixel = 0; pixel < NPixels; ++pixel) {
      peds[range][pixel]  = (float(rand()) / float(RAND_MAX)) * ((1 << RangeOffset) - 1);
      gains[range][pixel] = (float(rand()) / float(RAND_MAX)) * ((1 << RangeOffset) - 1); // What's appropriate here?
    }
    chkError(cudaMemcpy(peds_d,  peds[range].data(),  NPixels * sizeof(*peds_d),  cudaMemcpyDefault));
    chkError(cudaMemcpy(gains_d, gains[range].data(), NPixels * sizeof(*gains_d), cudaMemcpyDefault));
    peds_d  += NPixels;
    gains_d += NPixels;
  }

  auto& simDet = *static_cast<SimDet*>(m_det);
  simDet.calculateReference(peds, gains);

  return rc;
}

void EpixUHRsim::issuePhase2(TransitionId::Value tid)
{
  m_det->gpuDetector()->issuePhase2(tid);
}

float const* EpixUHRsim::referenceBuffers() const
{
  auto simDet = static_cast<SimDet*>(m_det);
  return simDet->referenceBuffers();
}

unsigned EpixUHRsim::referenceBufCnt() const
{
  auto simDet = static_cast<SimDet*>(m_det);
  return simDet->referenceBufCnt();
}

void EpixUHRsim::event(Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t count)
{
  //logging::info("Gpu::EpixUHRsim event");

  constexpr uint32_t lane{0}; // The lane is always 0 for GPU-enabled PGP devices
  DmaBuffer* buffer = &event->buffers[lane];
  auto size = buffer->size - sizeof(TimingHeader);
  constexpr auto panelSize{NPixels * sizeof(uint16_t)};
  if      (size  < panelSize)          dgram.xtc.damage.increase(Damage::MissingData);
  else if (size == m_pool->dmaSize())  dgram.xtc.damage.increase(Damage::Truncated);

  // @todo: Deal with prescaled raw data here?
}

// Instantiating the kernel template here puts the calibration in the same CUDA
// module as the kernel, so it inlines.  See ReaderKernels.cuh.
void EpixUHRsim::recordEvent(cudaStream_t           stream,
                             unsigned               blocks,
                             unsigned               threads,
                             const EventKernelArgs& args)
{
  PedGainCalib const calib{pedestals_d(),
                           gains_d(),
                           referenceBuffers(),
                           referenceBufCnt(),
                           rangeOffset(),
                           rangeBits()};
  _event<PedGainCalib><<<blocks, threads, 0, stream>>>(args, calib);
}

// The class factory

extern "C" Drp::Gpu::Detector* createDetector(Drp::Parameters& para, Drp::Gpu::MemPoolGpu& pool)
{
  return new EpixUHRsim(para, pool);
}
