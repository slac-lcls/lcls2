#include "EpixUHRemu.hh"

#include "GpuAsyncLib.hh"
#include "psdaq/service/EbDgram.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/DescData.hh"
#include "psalg/utils/SysLog.hh"

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Pds;
using namespace Drp::Gpu;
using json = nlohmann::json;

namespace Drp {
  class PGPEvent;
  namespace Gpu {

// The functionality of the Drp::Detector is need to set up each of the panels
// However, the data description and handling in the GPU case will be different,
// so we create a Drp Detector class to handle just the protion we need.
// Derive from Drp::XpmDetector so it can be made non-abstract.
class XpmDetector : public Drp::XpmDetector
{
public:
  XpmDetector(Parameters* para, MemPool* pool, unsigned len=100) : Drp::XpmDetector(para, pool, len) {}
  using Drp::XpmDetector::event;
  void event(Dgram&, const void* bufEnd, PGPEvent*, uint64_t count) override { /* Not used */ }
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


EpixUHRemu::EpixUHRemu(Parameters& para, MemPoolGpu& pool) :
  Drp::Gpu::Detector(&para, &pool)
{
  // Call common code to set up a vector of Drp::XpmDetectors
  _initialize<Drp::Gpu::XpmDetector>(para, pool);

  // Check there is enough space in the DMA buffers for this many pixels
  assert(NPixels <= (pool.dmaSize() - sizeof(DmaDsc) - sizeof(TimingHeader)) / sizeof(uint16_t));

  // Set up buffers
  pool.createCalibBuffers(m_dets.size(), NPixels);

  // Allocate space for the calibration constants for each panel
  m_peds_d.resize(m_dets.size());
  m_gains_d.resize(m_dets.size());
  for (unsigned i = 0; i < m_dets.size(); ++i) {
    chkError(cudaMalloc(&m_peds_d[i],  NGains * NPixels * sizeof(*m_peds_d[i])));
    chkError(cudaMalloc(&m_gains_d[i], NGains * NPixels * sizeof(*m_gains_d[i])));
  }
}

EpixUHRemu::~EpixUHRemu()
{
  printf("*** EpixUHRemu dtor 1\n");
  for (unsigned i = 0; i < m_dets.size(); ++i) {
    chkError(cudaFree(m_peds_d[i]));
    chkError(cudaFree(m_gains_d[i]));
  }
  printf("*** EpixUHRemu dtor 2\n");

  auto pool = m_pool->getAs<MemPoolGpu>();
  pool->destroyCalibBuffers();
  printf("*** EpixUHRemu dtor 3\n");
}

unsigned EpixUHRemu::configure(const std::string& config_alias, Xtc& xtc, const void* bufEnd)
{
  logging::info("Gpu::EpixUHRemu configure");
  unsigned rc = 0;

  // Configure the XpmDetector for each panel in turn
  // @todo: Do we really want to extend the Xtc for each panel, or does one speak for all?
  unsigned panel = 0;
  for (const auto& det : m_dets) {
    printf("*** Gpu::EpixUHRemu configure for %u start\n", panel);
    rc = det->configure(config_alias, xtc, bufEnd);
    printf("*** Gpu::EpixUHRemu configure for %u done: rc %d, sz %u\n", panel, rc, xtc.sizeofPayload());
    if (rc) {
      logging::error("Gpu::EpixUHRemu::configure failed for %s\n", m_params[panel].device);
      break;
    }
    ++panel;
  }

  Alg alg("raw", 0, 0, 0);
  NamesId namesId(nodeId, EventNamesIndex); // + panel);
  Names& names = *new(xtc, bufEnd) Names(bufEnd,
                                         m_para->detName.c_str(), alg,
                                         m_para->detType.c_str(), m_para->serNo.c_str(), namesId, m_para->detSegment);
  RawDef dataDef;
  names.add(xtc, bufEnd, dataDef);
  m_namesLookup[namesId] = NameIndex(names);

  logging::info("Gpu::EpixUHRemu configure: xtc size %u", xtc.sizeofPayload());

  return 0;
}

void EpixUHRemu::event(XtcData::Dgram& dgram, const void* bufEnd, PGPEvent*, uint64_t count)
{
  logging::info("Gpu::EpixUHRemu event");

  // @todo: Deal with prescaled raw or calibrated data for each panel here?
}

unsigned EpixUHRemu::beginrun(Xtc& xtc, const void* bufEnd, const json& runInfo)
{
  // Do beginRun for each panel in turn
  unsigned rc = 0;
  unsigned i = 0;
  for (const auto& det : m_dets) {
    rc = det->beginrun(xtc, bufEnd, runInfo);
    if (rc) {
      logging::error("Gpu::EpixUHRemu::beginrun failed for %s\n", m_params[i].device);
      break;
    }
  }

  // @todo: Fetch calibration constants for each panel and store them on the GPU
  std::vector<float> peds(NPixels, 0.0);
  std::vector<float> gains(NPixels, 1.0);
  for (unsigned i = 0; i < m_dets.size(); ++i) {
    auto peds_d  = m_peds_d[i];
    auto gains_d = m_gains_d[i];
    for (unsigned gn = 0; gn < NGains; ++gn) {
      chkError(cudaMemcpy(peds_d,  peds.data(),  NPixels * sizeof(*peds_d),  cudaMemcpyHostToDevice));
      chkError(cudaMemcpy(gains_d, gains.data(), NPixels * sizeof(*gains_d), cudaMemcpyHostToDevice));
      peds_d  += NPixels * sizeof(*peds_d);
      gains_d += NPixels * sizeof(*gains_d);
    }
  }

  return rc;
}

// This kernel performs the data calibration
static __global__ void _calibrate(float*   const        __restrict__ calibBuffers,
                                  const size_t                       calibBufsCnt,
                                  uint16_t const* const __restrict__ in,
                                  const unsigned&                    index,
                                  const unsigned                     panel,
                                  float* const         __restrict__  peds_,
                                  float* const         __restrict__  gains_)
{
  int pixel = blockIdx.x * blockDim.x + threadIdx.x;
  auto const __restrict__ out = &calibBuffers[index * calibBufsCnt + panel * EpixUHRemu::NPixels];

  const auto gainMask = (1 << EpixUHRemu::GainOffset) - 1;
  for (int i = pixel; i < EpixUHRemu::NPixels; i += blockDim.x * gridDim.x) {
    const auto gain     = (in[i] >> EpixUHRemu::GainOffset) & ((1 << EpixUHRemu::GainBits) - 1);
    const auto peds     = &peds_ [gain * EpixUHRemu::NPixels];
    const auto gains    = &gains_[gain * EpixUHRemu::NPixels];
    const auto data     = in[i] & gainMask;
    out[i] = (float(data) - peds[i]) * gains[i];
  }
}

// This routine records the graph that calibrates the data
void EpixUHRemu::recordGraph(cudaStream_t&         stream,
                             const unsigned&       index_d,
                             const unsigned        panel,
                             uint16_t const* const rawBuffer)
{
  auto nPanels = m_dets.size();

  // Check that panel is within range
  assert (panel < nPanels);

  int threads = 1024;
  int blocks  = (NPixels + threads-1) / threads; // @todo: Limit this?
  auto       pool         = m_pool->getAs<MemPoolGpu>();
  const auto peds         = m_peds_d[panel];
  const auto gains        = m_gains_d[panel];
  auto const calibBuffers = pool->calibBuffers_d();
  auto const calibBufsCnt = pool->calibBufsSize();
  _calibrate<<<blocks, threads, 0, stream>>>(calibBuffers, calibBufsCnt, rawBuffer, index_d, panel, peds, gains);
}

// The class factory

extern "C" Drp::Gpu::Detector* createDetector(Drp::Parameters& para, Drp::Gpu::MemPoolGpu& pool)
{
  return new EpixUHRemu(para, pool);
}
