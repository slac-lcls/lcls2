#include "EpixUHR3x2.hh"

#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Pds;
using namespace Drp::Gpu;
using json = nlohmann::json;

struct uhr3x2_domain{ static constexpr char const* name{"EpixUHR3x2"}; };
using uhr3x2_scoped_range = nvtx3::scoped_range_in<uhr3x2_domain>;


namespace Drp {
  class PGPEvent;
  namespace Gpu {

class RawDef : public VarDef
{
public:
  enum index
    {
      raw
    };

  RawDef()
  {
    Alg raw("raw", 0, 0, 0);
    NameVec.push_back({"raw", Name::UINT8, 1});
  }
};
  } // Gpu
} // Drp

EpixUHR3x2::EpixUHR3x2(Parameters& para, MemPoolGpu& pool) :
  Drp::Gpu::Detector(&para, &pool)
{
  // Use the CPU-side Drp::EpixUHR3x2 for panel setup, configuration and the
  // Python-driven bits.  The data description and handling in the GPU case are
  // different, so only that portion is borrowed.
  _initialize<Drp::EpixUHR3x2>(para, pool);

  // Check there is enough space in the DMA buffers for this many pixels.  The
  // AxiStream Batcher adds a header line plus a tail line per sub-frame, and
  // pads each sub-frame up to a line boundary, so allow for that.  Not an
  // assert(): the GPU DRP is built with NDEBUG, which would compile it out.
  constexpr size_t maxLineWidth{64};     // Widest AXI stream this can arrive on
  constexpr size_t batchOverhead{(1 + NumSubFrames) * maxLineWidth};
  constexpr size_t minDmaSize{NPixels * sizeof(uint16_t) + sizeof(TimingHeader) + batchOverhead};
  if (minDmaSize > pool.dmaSize()) {
    logging::critical("DMA buffer of %zu bytes is too small for %u pixels in %u sub-frames: need %zu",
                      pool.dmaSize(), NPixels, NumSubFrames, minDmaSize);
    abort();
  }

  // Set up buffers
  pool.createCalibBuffers(NPixels);

  // Allocate space for the calibration constants
  chkError(cudaMalloc(&m_pedsVec_d,  NRanges * NPixels * sizeof(*m_pedsVec_d)));
  chkError(cudaMalloc(&m_gainsVec_d, NRanges * NPixels * sizeof(*m_gainsVec_d)));
}

EpixUHR3x2::~EpixUHR3x2()
{
  auto pool = m_pool->getAs<MemPoolGpu>();
  if (m_gainsVec_d)  chkError(cudaFree(m_gainsVec_d));
  if (m_pedsVec_d)   chkError(cudaFree(m_pedsVec_d));

  pool->destroyCalibBuffers();
}

unsigned EpixUHR3x2::configure(const std::string& config_alias, Xtc& xtc, const void* bufEnd)
{
  logging::info("Gpu::EpixUHR3x2 configure");

  // Configure the CPU-side detector for the panel
  unsigned rc = m_det->configure(config_alias, xtc, bufEnd);
  if (rc) {
    logging::error("Gpu::EpixUHR3x2::configure failed for %s\n", m_para->device);
    return rc;
  }

  Alg alg("raw", 0, 0, 0);
  NamesId namesId(nodeId, EventNamesIndex);
  Names& names = *new(xtc, bufEnd) Names(bufEnd,
                                         m_para->detName.c_str(), alg,
                                         m_para->detType.c_str(), m_para->serNo.c_str(), namesId, m_para->detSegment);
  RawDef dataDef;
  names.add(xtc, bufEnd, dataDef);
  m_namesLookup[namesId] = NameIndex(names);

  logging::info("Gpu::EpixUHR3x2 configure: xtc size %u", xtc.sizeofPayload());

  return 0;
}

unsigned EpixUHR3x2::beginrun(Xtc& xtc, const void* bufEnd, const json& runInfo)
{
  unsigned rc = m_det->beginrun(xtc, bufEnd, runInfo);
  if (rc) {
    logging::error("Gpu::EpixUHR3x2::beginrun failed for %s\n", m_para->device);
    return rc;
  }

  // Load the calibration constants onto the GPU
  // @todo: Fetch calibration constants
  std::vector<float> peds(NPixels, 0.0);
  std::vector<float> gains(NPixels, 1.0);
  auto peds_d  = m_pedsVec_d;
  auto gains_d = m_gainsVec_d;
  for (unsigned range = 0; range < NRanges; ++range) {
    chkError(cudaMemcpy(peds_d,  peds.data(),  NPixels * sizeof(*peds_d),  cudaMemcpyDefault));
    chkError(cudaMemcpy(gains_d, gains.data(), NPixels * sizeof(*gains_d), cudaMemcpyDefault));
    peds_d  += NPixels;
    gains_d += NPixels;
  }

  return rc;
}

void EpixUHR3x2::event(Dgram& dgram, const void* bufEnd, PGPEvent* event, uint64_t count)
{
  constexpr uint32_t lane{0}; // The lane is always 0 for GPU-enabled PGP devices
  DmaBuffer* buffer = &event->buffers[lane];
  size_t size = buffer->size;

  // The batched payload is at least the pixel data plus the TimingHeader; the
  // batcher's own header, tails and line padding make it somewhat larger
  constexpr auto minEventSize{sizeof(TimingHeader) + NPixels * sizeof(uint16_t)};
  if      (size  < minEventSize)       dgram.xtc.damage.increase(Damage::MissingData);
  else if (size == m_pool->dmaSize())  dgram.xtc.damage.increase(Damage::Truncated);

  // @todo: Deal with prescaled raw for the panel here?
}

// The class factory

extern "C" Drp::Gpu::Detector* createDetector(Drp::Parameters& para, Drp::Gpu::MemPoolGpu& pool)
{
  return new EpixUHR3x2(para, pool);
}
